"""WeightNorm and execution tests, with synthetic complete trials only."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from rl2_reproduction.task3.ppo_learner import compute_losses as reference_losses
from rl2_reproduction.task3.sequence_batch import collate_trials, compute_gae
from rl2_reproduction.task3.trial_collector import TrialCollector
from rl2_reproduction.task3_aligned.model import WeightNormGRU, WeightNormVisualPolicy, WeightNormWeight
from rl2_reproduction.task3_aligned.learner import AlignedPPOLearner, compute_losses, replay, _pad_graph_batch


class TinyEnv:
    """Action-independent synthetic episodes, not an engine-validation fixture."""
    max_steps_per_episode = 250

    def __init__(self):
        self.resets = 0

    def image(self):
        return ((np.arange(3600).reshape(3, 30, 40) + 17*self.steps) % 256).astype(np.uint8)

    def reset_episode(self, task):
        self.limit = task['lengths'][self.resets % 2]
        self.resets += 1
        self.steps = 0
        return self.image(), {'pose': [64, 64, 0]}

    def step(self, action):
        self.steps += 1
        done = self.steps == self.limit
        return (None if done else self.image()), -.04+float(done), done, {'success': done}


def policy(hidden_dim=12):
    torch.manual_seed(82)
    return WeightNormVisualPolicy(hidden_dim)


def trials(model):
    collector = TrialCollector(TinyEnv(), model, torch.Generator().manual_seed(93))
    return [collector.collect({'task_id': 'a', 'lengths': [2, 3]}),
            collector.collect({'task_id': 'b', 'lengths': [1, 2]})]


def test_weightnorm_all_matrices_output_axes_initialization_and_unique_parameters():
    model = policy(256)
    weights = [module for module in model.modules() if isinstance(module, WeightNormWeight)]
    assert len(weights) == 10  # 2 CNN + embedding + fusion + 4 GRU + 2 heads.
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
    for weight in weights:
        torch.testing.assert_close(weight(), weight.v, atol=0, rtol=0)
        expected = torch.linalg.vector_norm(weight.v, dim=weight.norm_axes, keepdim=True)
        torch.testing.assert_close(weight.g, expected, atol=0, rtol=0)
    embedding = model.action_embedding.normalized_weight
    assert embedding.output_axis == 1 and embedding.g.shape == (1, 256)
    # One-hot dense output columns, rather than independent token-row scaling.
    with torch.no_grad():
        embedding.g[0, 5] *= 2
    torch.testing.assert_close(embedding()[:, 5], 2*embedding.v[:, 5])
    torch.testing.assert_close(embedding()[:, 4], embedding.v[:, 4])
    for gate in (*model.gru._h2zr.weight.chunk(2, 0), model.gru._h2hhat.weight):
        torch.testing.assert_close(gate @ gate.T, torch.eye(256), atol=2e-6, rtol=2e-5)
    assert all(torch.count_nonzero(p) == 0 for name, p in model.named_parameters() if name.endswith('bias'))
    assert model.gru._forget_bias == 1. and '_forget_bias' not in model.gru.state_dict()
    assert model.cnn(torch.zeros(1, 3, 30, 40)).shape == (1, 16, 5, 7)
    parameters = list(model.parameters())
    assert len(parameters) == len({id(p) for p in parameters})
    assert model.architecture['data_dependent_initialization'] is False


@pytest.mark.parametrize('length', [1, 6])
def test_visual_step_and_full_sequence_agree(length):
    model = policy()
    images = torch.randint(256, (2, length, 3, 30, 40), dtype=torch.uint8)
    actions = torch.randint(3, (2, length))
    rewards, dones = torch.randn(2, length), torch.zeros(2, length)
    logits, values, last = model.forward_sequence(images, actions, rewards, dones, torch.tensor([length]*2))
    hidden, step_logits, step_values = model.initial_state(2), [], []
    for t in range(length):
        lp, value, hidden = model.forward_step(images[:, t], actions[:, t], rewards[:, t], dones[:, t], hidden)
        step_logits.append(lp)
        step_values.append(value)
    torch.testing.assert_close(logits, torch.stack(step_logits, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(values, torch.stack(step_values, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(last, hidden, atol=2e-6, rtol=2e-5)


def test_cross_episode_backprop_and_b1_reset_only_cut_history():
    torch.manual_seed(17)
    gru = WeightNormGRU(9, 12)
    inputs = torch.randn(1, 6, 9, requires_grad=True)
    features, _ = gru(inputs)
    features[0, -1].square().sum().backward()
    assert inputs.grad[0, :3].abs().sum() > 0
    reset_inputs = inputs.detach().clone().requires_grad_()
    features, _ = gru(reset_inputs, reset_mask=torch.tensor([[True, False, False, True, False, False]]))
    features[0, -1].square().sum().backward()
    assert torch.count_nonzero(reset_inputs.grad[0, :3]) == 0
    assert reset_inputs.grad[0, 3:].abs().sum() > 0


def test_episode_two_visual_gradient_reaches_episode_one_pixels():
    model = policy()
    images = torch.randn(1, 6, 3, 30, 40, requires_grad=True)
    actions = torch.zeros(1, 6, dtype=torch.long)
    rewards, dones = torch.zeros(1, 6), torch.tensor([[0., 0., 0., 1., 0., 0.]])
    logits, _, _ = model.forward_sequence(images, actions, rewards, dones, torch.tensor([6]))
    logits[0, -1].square().sum().backward()
    assert images.grad[0, :3].abs().sum() > 0
    reset_images = images.detach().clone().requires_grad_()
    reset_mask = torch.tensor([[True, False, False, True, False, False]])
    logits, _, _ = model.forward_sequence(reset_images, actions, rewards, dones, torch.tensor([6]), reset_mask)
    logits[0, -1].square().sum().backward()
    assert torch.count_nonzero(reset_images.grad[0, :3]) == 0
    assert reset_images.grad[0, 3:].abs().sum() > 0


def test_collector_feedback_and_replay_cross_episode_are_preserved():
    model = policy()
    records = trials(model)
    record = records[0]
    assert record.length == 5 and record.arrays['episode_done'].tolist() == [False, True, False, False, True]
    assert record.arrays['prev_episode_done'][2] == 1
    assert record.arrays['prev_actions'][2] == record.arrays['actions'][1]
    assert record.arrays['prev_rewards'][2] == record.arrays['rewards'][1]
    assert record.trace[2]['hidden_norm_before'] == record.trace[1]['hidden_norm_after']
    learner = AlignedPPOLearner(model, torch.optim.AdamW(model.parameters()), {}, use_cuda_graph=False)
    assert max(learner._replay_check(collate_trials(records)).values()) < 1e-5


def test_static_masked_ppo_matches_original_objective_and_gradients():
    model = policy()
    reference = deepcopy(model)
    batch = compute_gae(collate_trials(trials(model)))
    valid = batch['mask']
    batch['old_log_probs'][valid] += torch.linspace(-.6, .6, int(valid.sum()))
    batch['advantages'][valid] = torch.linspace(-2, 2, int(valid.sum()))
    expected, actual = reference_losses(reference, batch), compute_losses(model, batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-7, rtol=2e-5)
    expected['loss'].backward()
    actual['loss'].backward()
    for actual_parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, atol=2e-7, rtol=3e-5)


def test_padding_poison_and_empty_graph_rows_do_not_change_valid_loss_or_state():
    model = policy()
    batch = compute_gae(collate_trials(trials(model)))
    padded = _pad_graph_batch(batch, 4, timesteps=9)
    poison = deepcopy(padded)
    invalid = ~poison['mask']
    for key, value in poison.items():
        if key == 'mask':
            continue
        if value.is_floating_point():
            value[invalid] = float('nan')
        elif value.dtype == torch.uint8:
            value[invalid] = 255
        elif value.dtype == torch.bool:
            value[invalid] = True
        else:
            value[invalid] = 999
    base_losses = compute_losses(model, batch)
    padded_losses = compute_losses(model, padded)
    poisoned_losses = compute_losses(model, poison)
    for key in base_losses:
        torch.testing.assert_close(base_losses[key], padded_losses[key], atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(padded_losses[key], poisoned_losses[key], atol=0, rtol=0)
    base_state = replay(model, batch)[2]
    padded_state, poisoned_state = replay(model, padded)[2], replay(model, poison)[2]
    torch.testing.assert_close(base_state, padded_state[:2], atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(padded_state, poisoned_state, atol=0, rtol=0)
    assert torch.count_nonzero(padded_state[2:]) == 0
    # Padding gradients also remain finite and unaltered.
    poisoned_losses['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cpu_complete_trial_update_changes_shared_network_and_unique_optimizer():
    model = policy(256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-5, weight_decay=0)
    learner = AlignedPPOLearner(model, optimizer, {'epochs': 2, 'minibatch_trials': 1}, use_cuda_graph=False)
    result = learner.update(trials(model))
    assert result['training_steps'] == 8 and result['trial_count'] == 2
    assert result['optimizer_steps'] in (2, 4) and all(result['gradients_nonzero'].values())
    assert result['cnn_parameter_l2_change'] > 0 and result['gru_parameter_l2_change'] > 0
    assert result['execution_backend'] == 'cpu_eager' and result['shared_optimizer_unique']
    assert len(optimizer.state) == len(list(model.parameters()))
    with pytest.raises(ValueError, match='exactly once'):
        AlignedPPOLearner(model, torch.optim.AdamW(model.cnn.parameters()), {}, use_cuda_graph=False)
    with pytest.raises(ValueError, match='non-CUDA'):
        AlignedPPOLearner(model, optimizer, {}, use_cuda_graph=True)
