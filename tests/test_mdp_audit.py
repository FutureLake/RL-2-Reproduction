"""Independent checks of RL2 trial semantics, not merely upstream parity."""

import numpy as np
import torch

from rl2_reproduction import benchmark
from rl2_reproduction.device_adapter import device_adapters


class _TenStepEpisodes:
    def __init__(self, batch):
        self.batch, self.t = batch, 0

    def reset(self):
        self.t = 0
        return np.zeros(self.batch, dtype=np.int64)

    def step(self, actions):
        self.t += 1
        return (np.full(self.batch, self.t % 10, dtype=np.int64),
                np.ones(self.batch, dtype=np.float32),
                np.full(self.batch, self.t % 10 == 0, dtype=np.float32))


class _TraceNetwork:
    def __init__(self, policy):
        self.policy = policy
        self.received = []

    def initial_state(self, batch):
        return torch.zeros(batch, 1)

    def __call__(self, curr_obs, prev_action, prev_reward, prev_done, prev_state):
        self.received.append({"state": prev_state.clone(), "done": prev_done.clone(),
                              "reward": prev_reward.clone(), "obs": curr_obs.clone()})
        batch = len(curr_obs)
        output = (torch.distributions.Categorical(logits=torch.zeros(batch, 5))
                  if self.policy else torch.full((batch,), 2.0))
        return output, prev_state + 1


def test_trial_credit_and_memory_cross_all_nine_episode_boundaries(monkeypatch):
    config = {"trials_per_update": 2, "meta_episode_len": 100,
              "discount_gamma": 1.0, "gae_lambda": 1.0}
    monkeypatch.setattr(benchmark, "make_tasks", lambda _config, count, seed: _TenStepEpisodes(count))
    policy, value = _TraceNetwork(True), _TraceNetwork(False)
    data, _ = benchmark.collect(config, policy, value, task_seed=1, device="cpu")
    # With gamma=lambda=1, every lambda-return must equal its full remaining
    # reward sum. In particular, episode boundaries cannot terminate credit.
    remaining = torch.arange(100, 0, -1, dtype=torch.float32).repeat(2, 1)
    torch.testing.assert_close(data["tdlam_rets"], remaining, atol=0, rtol=0)
    torch.testing.assert_close(data["advs"], remaining - 2, atol=0, rtol=0)
    assert data["advs"][0, -1] == -1  # Trial end has zero bootstrap.
    for net in (policy, value):
        for t in range(10, 100, 10):
            received = net.received[t]
            assert torch.all(received["state"] == t)
            assert torch.all(received["done"] == 1)
            assert torch.all(received["reward"] == 1)
            assert torch.all(received["obs"] == 0)
    benchmark.collect(config, policy, value, task_seed=2, device="cpu")
    assert torch.count_nonzero(policy.received[100]["state"]) == 0
    assert torch.count_nonzero(value.received[100]["state"]) == 0


def test_sequence_replay_keeps_terminal_action_and_reward_for_next_episode():
    data = {"obs": torch.tensor([[0, 1, 0, 1]]),
            "acs": torch.tensor([[2, 1, 0, 2]]),
            "rews": torch.tensor([[3.0, -4.0, 5.0, 6.0]]),
            "dones": torch.tensor([[0.0, 1.0, 0.0, 1.0]])}
    actual = benchmark.sequence_inputs(data)
    torch.testing.assert_close(actual["prev_action"], torch.tensor([[0, 2, 1, 0]]))
    torch.testing.assert_close(actual["prev_reward"], torch.tensor([[0.0, 3.0, -4.0, 5.0]]))
    torch.testing.assert_close(actual["prev_done"], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_actual_gru_backpropagates_reward_history_across_episode_done(make_system):
    _, policy, _ = make_system()
    rewards = torch.linspace(-0.6, 0.8, 20).reshape(1, 20).requires_grad_()
    done = torch.zeros(1, 20)
    done[:, 0] = done[:, 10] = 1
    inputs = {"curr_obs": (torch.arange(20) % 4).reshape(1, 20),
              "prev_action": (torch.arange(20) % 3).reshape(1, 20),
              "prev_reward": rewards, "prev_done": done}
    with device_adapters():
        distribution, _ = policy(**inputs, prev_state=policy.initial_state(1))
        # A decision in episode two must differentiate through rewards supplied
        # during episode one. This detects accidental detach/reset at done.
        distribution.logits[0, 11, 0].backward()
    assert rewards.grad is not None
    assert torch.isfinite(rewards.grad).all()
    assert rewards.grad[0, 8:10].abs().sum() > 1e-7
    assert torch.count_nonzero(rewards.grad[0, 12:]) == 0
