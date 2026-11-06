"""Unchanged PPO objective with full-trial replay and optional static CUDA graph.

Only execution is reorganized: all trials remain intact, reductions use valid
steps, one optimizer owns the shared network. Old Task 3 code is not modified.
"""
import math
import time
import torch
from torch.nn import functional as F

from rl2_reproduction.task3.sequence_batch import collate_trials, compute_gae, select_trials

LOSS_KEYS = ('loss', 'policy_loss', 'value_loss', 'entropy', 'kl', 'clip_fraction', 'ratio_mean')
GRAPH_FIELDS = ('images_uint8', 'prev_actions', 'prev_rewards', 'prev_episode_done', 'reset_mask', 'mask',
                'old_logits', 'old_log_probs', 'advantages', 'return_targets', 'actions')


def replay(policy, batch):
    return policy.static_forward_sequence(batch['images_uint8'], batch['prev_actions'], batch['prev_rewards'],
                                          batch['prev_episode_done'], batch['mask'], batch['reset_mask'])


def compute_losses(policy, batch, config=None):
    """Static-shape exact categorical PPO; sanitize padding before arithmetic."""
    config = config or {}
    if config.get('standardize_advantages', False):
        raise ValueError('The aligned run preserves standardize_advantages=false')
    logits, values, _ = replay(policy, batch)
    valid = batch['mask']
    weight = valid.to(logits.dtype)
    count = weight.sum().clamp_min(1)
    mean = lambda tensor: (torch.where(valid, tensor, torch.zeros_like(tensor)) * weight).sum() / count
    clean = lambda key: torch.where(valid, batch[key].detach(), torch.zeros_like(batch[key]))
    old_logits = torch.where(valid[..., None], batch['old_logits'].detach(), torch.zeros_like(batch['old_logits']))
    old_logp, advantage, targets, actions = (clean(k) for k in ('old_log_probs','advantages','return_targets','actions'))
    normalized = logits - logits.logsumexp(-1, keepdim=True)
    old_normalized = old_logits - old_logits.logsumexp(-1, keepdim=True)
    probs, old_probs = normalized.softmax(-1), old_normalized.softmax(-1)
    logp = normalized.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(logp - old_logp)
    epsilon = config.get('clip_epsilon', .2)
    policy_loss = -mean(torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage))
    value_loss = mean(F.huber_loss(values, targets, delta=1., reduction='none'))
    entropy = mean(-(probs * normalized.clamp_min(torch.finfo(logits.dtype).min)).sum(-1))
    kl_terms = old_probs * (old_normalized - normalized)
    kl_terms = torch.where(probs == 0, torch.full_like(kl_terms, float('inf')), kl_terms)
    kl_terms = torch.where(old_probs == 0, torch.zeros_like(kl_terms), kl_terms)
    kl = mean(kl_terms.sum(-1))
    total = policy_loss + config.get('value_coef', .5)*value_loss - config.get('entropy_coef', .01)*entropy
    return dict(loss=total, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy, kl=kl,
                clip_fraction=mean(((ratio-1).abs()>epsilon).float()), ratio_mean=mean(ratio))


def _pad_graph_batch(batch, batch_size, timesteps=500):
    """Pad whole trial rows, never truncate or split their time dimension."""
    b, t = batch['mask'].shape
    if b > batch_size or t > timesteps or not bool(batch['mask'].any()):
        raise ValueError('Invalid or empty complete-trial graph batch')
    result = {}
    for key in GRAPH_FIELDS:
        source = batch[key]
        target = torch.zeros((batch_size, timesteps, *source.shape[2:]), dtype=source.dtype, device=source.device)
        target[:b, :t].copy_(source)
        result[key] = target
    return result


class _CudaGraphLoss:
    """One stable graph, with dynamic mask/data buffers and persistent gradients.

    The owning learner executes warmup, capture, replay and AdamW on one stream.
    Capture does not include optimizer steps, RNG or host synchronization.
    """
    def __init__(self, policy, batch, config, stream):
        self.policy, self.config = policy, config
        self.buffers = {key: value.clone() for key, value in batch.items()}
        self.parameters = list(policy.parameters())
        for parameter in self.parameters:
            parameter.grad = torch.zeros_like(parameter)
        for _ in range(2):
            self._zero_grad()
            warmup_losses = compute_losses(policy, self.buffers, config)
            warmup_losses['loss'].backward()
            del warmup_losses
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(self.graph, stream=stream):
            self._zero_grad()
            self.losses = compute_losses(policy, self.buffers, config)
            self.losses['loss'].backward()
        stream.synchronize()
        self.capture_seconds = time.perf_counter() - started

    def _zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.zero_()

    def forward_backward(self, batch):
        for key, value in batch.items():
            self.buffers[key].copy_(value)
        self.graph.replay()
        return self.losses


class _CudaGraphForward:
    """No-grad full-sequence graph for replay checks and post-epoch exact KL."""
    @torch.no_grad()
    def __init__(self, policy, batch, stream):
        keys = ('images_uint8', 'prev_actions', 'prev_rewards', 'prev_episode_done', 'mask', 'reset_mask')
        self.buffers = {key: batch[key].clone() for key in keys}
        for _ in range(2):
            warmup_outputs = replay(policy, self.buffers)
            del warmup_outputs
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = replay(policy, self.buffers)
        stream.synchronize()
        self.capture_seconds = time.perf_counter() - started

    @torch.no_grad()
    def forward(self, batch):
        for key, buffer in self.buffers.items():
            buffer.copy_(batch[key])
        self.graph.replay()
        return self.outputs


class AlignedPPOLearner:
    def __init__(self, policy, optimizer, config, shuffle_generator=None, use_cuda_graph=True):
        self.policy, self.optimizer = policy, optimizer
        self.config = dict(config.get('ppo', config))
        self.shuffle_generator = shuffle_generator or torch.Generator(device='cpu').manual_seed(0)
        self.use_cuda_graph = bool(use_cuda_graph)
        if self.use_cuda_graph and policy.device.type != 'cuda':
            raise ValueError('CUDA graph explicitly requested for a non-CUDA policy')
        registered = [id(p) for group in optimizer.param_groups for p in group['params']]
        required = [id(p) for p in policy.parameters() if p.requires_grad]
        if len(registered) != len(set(registered)) or set(registered) != set(required):
            raise ValueError('The unique optimizer must register every trainable parameter exactly once')
        if self.config.get('standardize_advantages', False) or not self.config.get('full_trial_bptt', True):
            raise ValueError('Aligned execution preserves no advantage standardization and full trial BPTT')
        self.minibatch_size = int(self.config.get('minibatch_trials', 8))
        if self.minibatch_size <= 0 or int(self.config.get('epochs',4)) <= 0:
            raise ValueError('Positive epochs and minibatch_trials required')
        self.stream = torch.cuda.Stream(device=policy.device) if policy.device.type == 'cuda' else None
        self.graph_loss = None
        self.graph_forward = None

    def _device_batch(self, batch):
        return {key: value.to(self.policy.device) for key, value in batch.items()}

    @torch.no_grad()
    def _inference_replay(self, batch):
        if not self.use_cuda_graph:
            return replay(self.policy, batch)
        padded = _pad_graph_batch(batch, self.minibatch_size)
        if self.graph_forward is None:
            self.graph_forward = _CudaGraphForward(self.policy, padded, self.stream)
        logits, values, state = self.graph_forward.forward(padded)
        b, t = batch['mask'].shape
        return logits[:b, :t], values[:b, :t], state[:b]

    @torch.no_grad()
    def _replay_check(self, batch, tolerance=1e-4):
        errors = dict(max_logprob_error=0., max_value_error=0., max_logit_error=0.)
        for start in range(0, len(batch['lengths']), self.minibatch_size):
            mini = self._device_batch(select_trials(batch, slice(start,start+self.minibatch_size)))
            logits, values, _ = self._inference_replay(mini)
            valid = mini['mask']
            logp = torch.distributions.Categorical(logits=logits[valid]).log_prob(mini['actions'][valid])
            measured = dict(max_logprob_error=float((logp-mini['old_log_probs'][valid]).abs().max()),
                            max_value_error=float((values[valid]-mini['old_values'][valid]).abs().max()),
                            max_logit_error=float((logits[valid]-mini['old_logits'][valid]).abs().max()))
            errors = {key:max(errors[key], measured[key]) for key in errors}
        if max(errors.values()) > tolerance:
            raise RuntimeError(f'CPU actor/full-trial replay mismatch: {errors}')
        return errors

    @torch.no_grad()
    def _full_kl(self, batch):
        weighted, count = 0., 0
        for start in range(0, len(batch['lengths']), self.minibatch_size):
            mini = self._device_batch(select_trials(batch, slice(start,start+self.minibatch_size)))
            steps = int(mini['mask'].sum())
            logits, _, _ = self._inference_replay(mini)
            distribution = torch.distributions.Categorical(logits=logits[mini['mask']])
            old_distribution = torch.distributions.Categorical(logits=mini['old_logits'][mini['mask']])
            weighted += float(torch.distributions.kl_divergence(old_distribution, distribution).mean()) * steps
            count += steps
        return weighted/count

    def update(self, trials):
        """One on-policy update. Graph failures are explicit errors, never fallback."""
        if self.stream is None:
            return self._update(trials)
        self.stream.wait_stream(torch.cuda.current_stream(self.policy.device))
        try:
            with torch.cuda.stream(self.stream):
                result = self._update(trials)
        finally:
            torch.cuda.current_stream(self.policy.device).wait_stream(self.stream)
        return result

    def _update(self, trials):
        start_time = time.perf_counter()
        config, policy = self.config, self.policy
        policy.train()
        # Entire rollout remains in host memory; only whole-trial minibatches
        # and chunked replay/KL move to CUDA, bounding activation memory.
        batch = compute_gae(collate_trials(trials, 'cpu'), config.get('gamma',.99), config.get('gae_lambda',.99))
        if int(batch['lengths'].max()) > 500:
            raise ValueError('Expected exactly two episodes with at most 250 actions each')
        new_forward_graph = self.use_cuda_graph and self.graph_forward is None
        consistency = self._replay_check(batch)
        groups = {name:list(module.parameters()) for name,module in (
            ('cnn',policy.cnn),('gru',policy.gru),('policy_head',policy.policy_head),('value_head',policy.value_head))}
        before = {name:[p.detach().clone() for p in params] for name,params in groups.items()}
        records, epoch_kls = [], []
        new_capture_seconds = 0.
        for epoch in range(config.get('epochs',4)):
            order = torch.randperm(len(trials), generator=self.shuffle_generator)
            for indices in order.split(self.minibatch_size):
                host = select_trials(batch, indices)
                valid_steps = int(host['mask'].sum())
                mini = self._device_batch(host)
                if self.use_cuda_graph:
                    padded = _pad_graph_batch(mini, self.minibatch_size)
                    if self.graph_loss is None:
                        self.graph_loss = _CudaGraphLoss(policy, padded, config, self.stream)
                        new_capture_seconds = self.graph_loss.capture_seconds
                    losses = self.graph_loss.forward_backward(padded)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses = compute_losses(policy, mini, config)
                    losses['loss'].backward()
                loss_values = {key:float(value.detach()) for key,value in losses.items()}
                if not all(math.isfinite(value) for value in loss_values.values()):
                    raise FloatingPointError(f'Non-finite PPO objectives: {loss_values}')
                if any(parameter.grad is None for parameter in policy.parameters()):
                    raise FloatingPointError('Missing shared-network gradient')
                gradient = {name+'_grad_norm':float(torch.stack([p.grad.detach().square().sum() for p in params]).sum().sqrt())
                            for name,params in groups.items()}
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),config.get('max_grad_norm',.5),error_if_nonfinite=True)
                self.optimizer.step()
                finite = torch.stack([torch.isfinite(p).all() for p in policy.parameters()]).all()
                if not bool(finite):
                    raise FloatingPointError('Non-finite updated parameters')
                records.append({**loss_values,**gradient,'global_grad_norm_before_clip':float(grad_norm),
                                'valid_steps':valid_steps,'epoch':epoch+1})
            kl = self._full_kl(batch)
            epoch_kls.append(kl)
            if kl > config.get('epoch_stop_kl',.03):
                break
        changes = {name+'_parameter_l2_change':float(torch.stack([(p.detach()-old).square().sum() for p,old in zip(params,before[name])]).sum().sqrt())
                   for name,params in groups.items()}
        weight = sum(row['valid_steps'] for row in records)
        mean = {key:sum(row[key]*row['valid_steps'] for row in records)/weight for key in LOSS_KEYS}
        return {**mean,**consistency,**changes,'training_steps':int(batch['mask'].sum()),'trial_count':len(trials),
                'optimizer_steps':len(records),'epochs_completed':len(epoch_kls),'epoch_kls':epoch_kls,'minibatches':records,
                'gradients_nonzero':{name:any(row[name+'_grad_norm']>0 for row in records) for name in groups},
                'shared_optimizer_unique':True, 'execution_backend':'cuda_graph' if self.use_cuda_graph else str(policy.device)+'_eager',
                'cuda_graph_capture_seconds_this_update':new_capture_seconds,
                'cuda_graph_forward_capture_seconds_this_update':self.graph_forward.capture_seconds if new_forward_graph else 0.,
                'learner_seconds':time.perf_counter()-start_time,
                'full_trial_bptt':True,'graph_shape':[self.minibatch_size,500] if self.use_cuda_graph else None}
