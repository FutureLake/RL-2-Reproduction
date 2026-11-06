"""Masked full-trial PPO with one shared-network optimizer."""

import torch
from torch.nn import functional as F
from .sequence_batch import collate_trials, compute_gae, select_trials


def replay(policy, batch):
    return policy.forward_sequence(batch["images_uint8"], batch["prev_actions"], batch["prev_rewards"],
                                   batch["prev_episode_done"], batch["lengths"], batch["reset_mask"])


def compute_losses(policy, batch, config=None):
    config = config or {}
    logits, values, _ = replay(policy, batch)
    valid = batch["mask"]
    # Index before calculating losses: invalid padding may contain NaN or invalid actions.
    distribution = torch.distributions.Categorical(logits=logits[valid])
    old_logits = batch["old_logits"][valid].detach()
    old_distribution = torch.distributions.Categorical(logits=old_logits)
    old_logp = batch["old_log_probs"][valid].detach()
    advantage = batch["advantages"][valid].detach()
    if config.get("standardize_advantages", False):
        raise ValueError("v2 fixes standardize_advantages=false; changing this needs an explicit protocol revision")
    ratio = torch.exp(distribution.log_prob(batch["actions"][valid]) - old_logp)
    epsilon = config.get("clip_epsilon", 0.2)
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    value_loss = F.huber_loss(values[valid], batch["return_targets"][valid].detach(), delta=1.0)
    entropy = distribution.entropy().mean()
    total = policy_loss + config.get("value_coef", 0.5) * value_loss - config.get("entropy_coef", 0.01) * entropy
    return {"loss": total, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
            "kl": torch.distributions.kl_divergence(old_distribution, distribution).mean(),
            "clip_fraction": ((ratio - 1).abs() > epsilon).float().mean(), "ratio_mean": ratio.mean()}


@torch.no_grad()
def replay_check(policy, batch, tolerance=1e-4):
    logits, values, _ = replay(policy, batch)
    valid = batch["mask"]
    logp = torch.distributions.Categorical(logits=logits[valid]).log_prob(batch["actions"][valid])
    errors = {"max_logprob_error": float((logp - batch["old_log_probs"][valid]).abs().max()),
              "max_value_error": float((values[valid] - batch["old_values"][valid]).abs().max()),
              "max_logit_error": float((logits[valid] - batch["old_logits"][valid]).abs().max())}
    if max(errors.values()) > tolerance:
        raise RuntimeError(f"Rollout/full-trial replay mismatch: {errors}")
    return errors


class PPOLearner:
    def __init__(self, policy, optimizer, config, shuffle_generator=None):
        self.policy, self.optimizer = policy, optimizer
        self.config = config.get("ppo", config)
        self.shuffle_generator = shuffle_generator or torch.Generator(device="cpu").manual_seed(0)
        registered = [id(p) for group in optimizer.param_groups for p in group["params"]]
        required = [id(p) for p in policy.parameters() if p.requires_grad]
        if len(registered) != len(set(registered)) or set(registered) != set(required):
            raise ValueError("The unique optimizer must register every trainable model parameter exactly once")

    def update(self, trials):
        config = self.config
        self.policy.train()
        batch = compute_gae(collate_trials(trials, self.policy.device), config.get("gamma", 0.99), config.get("gae_lambda", 0.99))
        consistency = replay_check(self.policy, batch)
        groups = {name: list(module.parameters()) for name, module in
                  (("cnn", self.policy.cnn), ("gru", self.policy.gru), ("policy_head", self.policy.policy_head), ("value_head", self.policy.value_head))}
        before = {name: [p.detach().clone() for p in params] for name, params in groups.items()}
        records, epoch_kls = [], []
        minibatch_size = config.get("minibatch_trials", 8)
        if minibatch_size <= 0 or config.get("epochs", 4) <= 0:
            raise ValueError("PPO epochs and minibatch_trials must be positive")
        for epoch in range(config.get("epochs", 4)):
            order = torch.randperm(len(trials), generator=self.shuffle_generator)
            for indices in order.split(minibatch_size):
                minibatch = select_trials(batch, indices.to(self.policy.device))
                losses = compute_losses(self.policy, minibatch, config)
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Non-finite PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                gradient = {}
                for name, params in groups.items():
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                        raise FloatingPointError(f"Missing/non-finite {name} gradient")
                    gradient[name + "_grad_norm"] = float(torch.stack([p.grad.detach().square().sum() for p in params]).sum().sqrt())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.get("max_grad_norm", 0.5), error_if_nonfinite=True)
                self.optimizer.step()
                if any(not torch.isfinite(p).all() for p in self.policy.parameters()):
                    raise FloatingPointError("Non-finite updated parameters")
                records.append({**{key: float(value.detach()) for key, value in losses.items()}, **gradient,
                                "global_grad_norm_before_clip": float(grad_norm), "valid_steps": int(minibatch["mask"].sum()), "epoch": epoch + 1})
            with torch.no_grad():
                # Full batch KL, weighted by valid steps rather than padded length or trials.
                kl = float(compute_losses(self.policy, batch, config)["kl"])
            epoch_kls.append(kl)
            if kl > config.get("epoch_stop_kl", 0.03):
                break
        changes = {name + "_parameter_l2_change": float(torch.stack([(p.detach() - old).square().sum() for p, old in zip(params, before[name])]).sum().sqrt())
                   for name, params in groups.items()}
        weight = sum(row["valid_steps"] for row in records)
        mean = {key: sum(row[key] * row["valid_steps"] for row in records) / weight
                for key in ("loss", "policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "ratio_mean")}
        return {**mean, **consistency, **changes, "training_steps": int(batch["mask"].sum()),
                "trial_count": len(trials), "optimizer_steps": len(records), "epochs_completed": len(epoch_kls),
                "epoch_kls": epoch_kls, "minibatches": records,
                "gradients_nonzero": {name: any(row[name + "_grad_norm"] > 0 for row in records) for name in groups},
                "shared_optimizer_unique": True}
