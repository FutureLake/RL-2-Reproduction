"""Task 3 numerical tests use explicit fake episodes, never claim engine validation."""

from copy import deepcopy
import io
import random
import numpy as np
import pytest
import torch

from rl2_reproduction.task3.visual_policy import VisualPolicy, Task3GRU
from rl2_reproduction.task3.trial_collector import TrialCollector
from rl2_reproduction.task3.sequence_batch import collate_trials, compute_gae
from rl2_reproduction.task3.ppo_learner import PPOLearner, compute_losses, replay_check
from rl2_reproduction.task3.evaluate import evaluate_tasks


class FakeEpisodeEnv:
    """Deterministic short episodes for testing the collector, not ViZDoom."""
    max_steps_per_episode = 250

    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.finished = False

    def _image(self):
        values = np.arange(3 * 30 * 40, dtype=np.uint16).reshape(3, 30, 40)
        return ((values + self.steps * 17) % 256).astype(np.uint8)

    def reset_episode(self, task):
        self.task = task
        self.limit = task.get("length", task.get("lengths", [2, 3])[self.resets % 2])
        self.resets += 1
        self.steps = 0
        self.finished = False
        return self._image(), {"pose": [64, 64, 0]}

    def step(self, action):
        assert not self.finished, "Collector stepped after terminal"
        self.steps += 1
        self.finished = self.steps == self.limit
        success = self.finished and self.task.get("success", True)
        reward = -0.04 + float(success) - 0.001 * float(action == 2)
        return (None if self.finished else self._image()), reward, self.finished, {
            "pose": [64 + self.steps, 64, 0], "success": success, "collided": action == 2, "distance": 1.0}


def _policy(hidden_dim=12):
    torch.manual_seed(823)
    return VisualPolicy(hidden_dim)


def _trials(policy, mode="rl2"):
    collector = TrialCollector(FakeEpisodeEnv(), policy, torch.Generator().manual_seed(92))
    return [collector.collect({"task_id": "a", "lengths": [2, 3]}, mode=mode),
            collector.collect({"task_id": "b", "lengths": [1, 2]}, mode=mode)]


@pytest.mark.parametrize("sequence", [False, True])
def test_task3_gru_matches_pinned_upstream_outputs_and_gradients(sequence):
    from rl2.agents.architectures.gru import GRU
    torch.manual_seed(7)
    original = GRU(9, 12)
    adapted = Task3GRU(9, 12)
    adapted.load_state_dict(original.state_dict())
    shape = (2, 4, 9) if sequence else (2, 9)
    x = torch.randn(shape, requires_grad=True)
    adapted_x = x.detach().clone().requires_grad_()
    state = torch.randn(2, 12)
    expected, last = original(x, state)
    actual, adapted_last = adapted(adapted_x, state)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(adapted_last, last, atol=2e-6, rtol=2e-5)
    (expected.square().mean() + last.mean()).backward()
    (actual.square().mean() + adapted_last.mean()).backward()
    torch.testing.assert_close(adapted_x.grad, x.grad, atol=1e-5, rtol=1e-4)
    for name, p in original.named_parameters():
        torch.testing.assert_close(dict(adapted.named_parameters())[name].grad, p.grad, atol=1e-5, rtol=1e-4)


def test_full_architecture_and_per_gate_initialization():
    policy = _policy(256)
    assert policy.architecture["parameter_count"] == 616404
    assert policy.cnn(torch.zeros(1, 3, 30, 40)).shape == (1, 16, 5, 7)
    for gate in (*policy.gru._h2zr.weight.chunk(2, dim=0), policy.gru._h2hhat.weight):
        torch.testing.assert_close(gate @ gate.T, torch.eye(256), atol=2e-6, rtol=2e-5)
    assert policy.gru._forget_bias == 1.0
    assert "_forget_bias" not in policy.gru.state_dict()
    assert all(torch.count_nonzero(p) == 0 for name, p in policy.named_parameters() if name.endswith("bias") or name.endswith("._b"))


@pytest.mark.parametrize("length", [1, 4])
def test_visual_step_sequence_shapes_and_replay_match(length):
    policy = _policy()
    images = torch.randint(0, 256, (1, length, 3, 30, 40), dtype=torch.uint8)
    actions = torch.randint(0, 3, (1, length))
    rewards, dones = torch.randn(1, length), torch.zeros(1, length)
    logits, values, state = policy.forward_sequence(images, actions, rewards, dones, torch.tensor([length]))
    assert logits.shape == (1, length, 3) and values.shape == (1, length)
    h, step_logits, step_values = policy.initial_state(1), [], []
    for t in range(length):
        lp, value, h = policy.forward_step(images[:, t], actions[:, t], rewards[:, t], dones[:, t], h)
        step_logits.append(lp)
        step_values.append(value)
    torch.testing.assert_close(logits, torch.stack(step_logits, dim=1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(values, torch.stack(step_values, dim=1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state, h, atol=2e-6, rtol=2e-5)


def test_collector_exact_two_early_success_episodes_and_feedback():
    policy, env = _policy(), FakeEpisodeEnv()
    collector = TrialCollector(env, policy)
    trial = collector.collect({"task_id": "short", "lengths": [2, 3]})
    assert trial.length == 5 and env.resets == 2
    assert trial.arrays["episode_done"].tolist() == [False, True, False, False, True]
    assert trial.arrays["trial_done"].tolist() == [False] * 4 + [True]
    assert trial.arrays["prev_episode_done"].tolist() == [0, 0, 1, 0, 0]
    assert trial.arrays["prev_rewards"][2] == trial.arrays["rewards"][1]
    assert trial.arrays["prev_actions"][2] == trial.arrays["actions"][1]
    assert trial.trace[2]["hidden_norm_before"] == trial.trace[1]["hidden_norm_after"]
    assert trial.trace[0]["hidden_norm_before"] == 0
    another = collector.collect({"task_id": "next", "length": 1})
    assert another.trace[0]["hidden_norm_before"] == 0
    assert another.arrays["prev_rewards"][0] == another.arrays["prev_episode_done"][0] == 0
    assert env.resets == 4
    replay_check(policy, collate_trials([trial, another]))


def test_collector_complete_timeout_trials_and_b1_reset():
    policy = _policy()
    collector = TrialCollector(FakeEpisodeEnv(), policy)
    trial = collector.collect({"task_id": "timeouts", "length": 250, "success": False}, mode="b1")
    assert trial.length == 500
    assert trial.arrays["episode_done"].sum() == 2
    assert trial.arrays["reset_mask"][[0, 250]].all()
    assert trial.trace[250]["hidden_norm_before"] == 0
    assert trial.arrays["prev_episode_done"][250] == 1
    assert trial.arrays["prev_rewards"][250] == trial.arrays["rewards"][249]
    replay_check(policy, collate_trials([trial]))


def test_gae_hand_calculation_crosses_episode_but_stops_at_trial():
    batch = collate_trials(_trials(_policy()))
    batch["rewards"][batch["mask"]] = 1
    batch["old_values"][batch["mask"]] = 2
    compute_gae(batch, 1.0, 1.0)
    torch.testing.assert_close(batch["return_targets"], torch.tensor([[5., 4., 3., 2., 1.], [3., 2., 1., 0., 0.]]))
    assert batch["advantages"][0, -1] == -1
    # A nontrivial gamma/lambda calculation at the first episode boundary.
    compute_gae(batch, 0.9, 0.5)
    expected = torch.zeros(5)
    future_adv, future_value = 0., 0.
    for t in reversed(range(5)):
        future_adv = 1 + .9 * future_value - 2 + .9 * .5 * future_adv
        expected[t] = future_adv
        future_value = 2.
    torch.testing.assert_close(batch["advantages"][0], expected)


def test_episode_two_gradient_crosses_history_and_b1_cuts_only_memory():
    torch.manual_seed(17)
    gru = Task3GRU(9, 12)
    inputs = torch.randn(1, 6, 9, requires_grad=True)
    features, _ = gru(inputs)
    features[0, 5].square().sum().backward()
    assert inputs.grad[0, :3].abs().sum() > 0
    reset_inputs = inputs.detach().clone().requires_grad_()
    mask = torch.tensor([[True, False, False, True, False, False]])
    features, _ = gru(reset_inputs, reset_mask=mask)
    features[0, 5].square().sum().backward()
    assert torch.count_nonzero(reset_inputs.grad[0, :3]) == 0
    assert reset_inputs.grad[0, 3:].abs().sum() > 0


def test_padding_cannot_change_loss_kl_entropy_targets_or_final_state():
    policy = _policy()
    batch = compute_gae(collate_trials(_trials(policy)))
    changed = deepcopy(batch)
    invalid = ~changed["mask"]
    for key, value in changed.items():
        if key in ("lengths", "mask"):
            continue
        if value.is_floating_point():
            value[invalid] = float("nan")
        elif value.dtype == torch.uint8:
            value[invalid] = 255
        elif value.dtype != torch.bool:
            value[invalid] = 99
        else:
            value[invalid] = ~value[invalid]
    expected, actual = compute_losses(policy, batch), compute_losses(policy, changed)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    arguments = lambda b: (b["images_uint8"], b["prev_actions"], b["prev_rewards"], b["prev_episode_done"], b["lengths"], b["reset_mask"])
    torch.testing.assert_close(policy.forward_sequence(*arguments(batch))[2], policy.forward_sequence(*arguments(changed))[2], atol=0, rtol=0)
    compute_gae(changed)
    torch.testing.assert_close(changed["return_targets"][batch["mask"]], batch["return_targets"][batch["mask"]])


def test_ppo_clipping_huber_and_old_targets_are_detached():
    policy = _policy()
    batch = compute_gae(collate_trials(_trials(policy)))
    valid = batch["mask"]
    count = int(valid.sum())
    batch["old_log_probs"][valid] += torch.linspace(-0.6, 0.6, count)
    batch["advantages"][valid] = torch.linspace(-2.0, 2.0, count)
    for key in ("old_log_probs", "old_logits", "old_values", "advantages", "return_targets"):
        batch[key].requires_grad_()
    losses = compute_losses(policy, batch)
    assert losses["clip_fraction"] > 0
    logits, values, _ = policy.forward_sequence(batch["images_uint8"], batch["prev_actions"], batch["prev_rewards"],
                                                 batch["prev_episode_done"], batch["lengths"], batch["reset_mask"])
    logp = torch.distributions.Categorical(logits=logits[valid]).log_prob(batch["actions"][valid])
    ratios = (logp - batch["old_log_probs"][valid]).exp()
    expected_policy = -torch.minimum(ratios * batch["advantages"][valid], ratios.clamp(.8, 1.2) * batch["advantages"][valid]).mean()
    error = (values[valid] - batch["return_targets"][valid]).abs()
    expected_value = torch.where(error < 1, .5 * error.square(), error - .5).mean()
    torch.testing.assert_close(losses["policy_loss"], expected_policy)
    torch.testing.assert_close(losses["value_loss"], expected_value)
    losses["loss"].backward()
    for key in ("old_log_probs", "old_logits", "old_values", "advantages", "return_targets"):
        assert batch[key].grad is None


def test_full_model_cnn_gru_gradients_updates_single_optimizer_and_roundtrip():
    policy = _policy(256)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, eps=1e-5, weight_decay=0)
    learner = PPOLearner(policy, optimizer, {"epochs": 2, "minibatch_trials": 1})
    stats = learner.update(_trials(policy))
    assert stats["training_steps"] == 8
    assert stats["optimizer_steps"] in (2, 4)
    assert all(stats["gradients_nonzero"].values())
    assert stats["cnn_parameter_l2_change"] > 0 and stats["gru_parameter_l2_change"] > 0
    memory = io.BytesIO()
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict()}, memory)
    memory.seek(0)
    state = torch.load(memory, weights_only=False)
    restored = _policy(256)
    restored.load_state_dict(state["model"])
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4, eps=1e-5, weight_decay=0)
    new_optimizer.load_state_dict(state["optimizer"])
    batch = collate_trials(_trials(policy))
    replay_check(restored, batch)
    assert set(new_optimizer.state_dict()["state"]) == set(optimizer.state_dict()["state"])
    assert len(optimizer.state) == len(list(policy.parameters()))
    with pytest.raises(ValueError, match="exactly once"):
        PPOLearner(policy, torch.optim.AdamW(policy.cnn.parameters()), {})


def test_evaluation_freezes_weights_rng_and_shares_episode_one_prefix():
    policy = _policy()
    before = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    tasks = [{"task_id": "eval0", "length": 3}, {"task_id": "eval1", "length": 2}]
    env = FakeEpisodeEnv()
    result = evaluate_tasks(env, policy, tasks, action_seed=105)
    assert env.resets == 6  # One episode-one prefix, then two episode-two branches per task.
    assert result["same_episode1_prefix"] and result["weights_frozen"] and result["training_rng_preserved"]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(torch.equal(before[k], p) for k, p in policy.state_dict().items())
    for row, task in zip(result["trials"], tasks):
        length = task["length"]
        assert row["trace"][:length] == row["reset_trace"][:length]
        assert row["episode_summaries"][0] == row["reset_episode_summaries"][0]
        assert row["trace"][length]["prev_reward"] == row["reset_trace"][length]["prev_reward"]
        assert row["reset_trace"][length]["hidden_norm_before"] == 0
    assert result == evaluate_tasks(FakeEpisodeEnv(), policy, tasks, action_seed=105)


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lvalue, rvalue in zip(left, right):
            _assert_nested_equal(lvalue, rvalue)
    else:
        assert left == right


def test_update_boundary_checkpoint_exactly_resumes_second_update_and_all_rng(tmp_path):
    from rl2_reproduction.task3.checkpoint import save_checkpoint, load_checkpoint
    from rl2_reproduction.task3.utils import configure, capture_rng
    config = {"protocol": "numeric-test", "ppo": {"epochs": 2, "minibatch_trials": 1}}
    hashes = {"manifest": "test-fixture-manifest", "code": "test-fixture-code"}

    def build():
        configure(31)
        model = VisualPolicy(12)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-5, weight_decay=0)
        task_rng = np.random.default_rng(32)
        action_rng = torch.Generator().manual_seed(33)
        shuffle_rng = torch.Generator().manual_seed(34)
        learner = PPOLearner(model, optimizer, config, shuffle_rng)
        collector = TrialCollector(FakeEpisodeEnv(), model, action_rng)
        return model, optimizer, task_rng, action_rng, shuffle_rng, learner, collector

    def update(system):
        model, optimizer, task_rng, action_rng, shuffle_rng, learner, collector = system
        # Exercise every stored global RNG in addition to separate sampling streams.
        globals_sample = random.random(), float(np.random.random()), torch.rand(2)
        tasks = [{"task_id": str(int(task_rng.integers(0, 100000))), "length": int(task_rng.integers(1, 4))} for _ in range(2)]
        trials = [collector.collect(task) for task in tasks]
        stats = learner.update(trials)
        return globals_sample, tasks, trials, stats

    continuous = build()
    first = update(continuous)
    model, optimizer, task_rng, action_rng, shuffle_rng, _, _ = continuous
    save_checkpoint(tmp_path / "boundary.pt", model, optimizer, config, hashes, 1, first[3]["training_steps"], task_rng, action_rng, shuffle_rng)
    expected = update(continuous)
    expected_model, expected_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    expected_rng = capture_rng()
    expected_streams = deepcopy(task_rng.bit_generator.state), action_rng.get_state(), shuffle_rng.get_state()

    restored = build()
    new_model, new_optimizer, new_task_rng, new_action_rng, new_shuffle_rng, _, _ = restored
    state = load_checkpoint(tmp_path / "boundary.pt", new_model, new_optimizer, config, hashes,
                            new_task_rng, new_action_rng, new_shuffle_rng, restore_random=True)
    assert state["update"] == 1 and state["boundary"] == "complete_update"
    actual = update(restored)
    _assert_nested_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    for real, wanted in zip(actual[2], expected[2]):
        _assert_nested_equal(real.arrays, wanted.arrays)
        assert real.trace == wanted.trace
    _assert_nested_equal(actual[3], expected[3])
    _assert_nested_equal(new_model.state_dict(), expected_model)
    _assert_nested_equal(new_optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(capture_rng(), expected_rng)
    _assert_nested_equal((new_task_rng.bit_generator.state, new_action_rng.get_state(), new_shuffle_rng.get_state()), expected_streams)
    with pytest.raises(ValueError, match="hashes differ"):
        load_checkpoint(tmp_path / "boundary.pt", new_model, hashes={"manifest": "changed"})
