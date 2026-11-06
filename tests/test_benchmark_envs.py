"""Distribution, trial-boundary and isolation checks for batched benchmarks."""

import random

import numpy as np
import pytest
import torch

from rl2_reproduction.benchmark_envs import VectorTasks, evaluate


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_task_parameters_independent_of_rollout_and_repeatable(environment):
    first = VectorTasks(environment, 7, 10, 5, 3, 843)
    second = VectorTasks(environment, 7, 10, 5, 3, 843)
    original = first.task_fingerprint
    assert original == second.task_fingerprint
    for t in range(5):
        first.step(np.full(7, t % 5, dtype=np.int64))
    assert first.task_fingerprint == original
    assert first.task_fingerprints == second.task_fingerprints
    if environment == "bandit":
        assert first.payout_probabilities.shape == (7, 5)
        assert np.all((first.payout_probabilities >= 0) & (first.payout_probabilities <= 1))
    else:
        assert first.reward_means.shape == (7, 10, 5)
        assert first.transition_probabilities.shape == (7, 5, 10, 10)
        np.testing.assert_allclose(first.transition_probabilities.sum(-1), 1, atol=1e-12)
    different = VectorTasks(environment, 7, 10, 5, 3, 844)
    assert original != different.task_fingerprint


def test_mdp_uses_previous_state_reward_and_resets_only_episode_state():
    env = VectorTasks("tabular_mdp", 2, 3, 2, 2, 91)
    env.reward_means.fill(0)
    env.reward_means[:, 0, :] = 7
    env.reward_means[:, 1, :] = 11
    env.transition_probabilities.fill(0)
    env.transition_probabilities[..., 1] = 1
    before = env.task_fingerprint
    env.set_noise_schedule(np.full((2, 2), 0.5), np.zeros((2, 2)))
    obs, reward, done = env.step(np.zeros(2, dtype=np.int64))
    np.testing.assert_array_equal(obs, [1, 1])
    np.testing.assert_array_equal(reward, [7, 7])
    np.testing.assert_array_equal(done, [0, 0])
    obs, reward, done = env.step(np.ones(2, dtype=np.int64))
    np.testing.assert_array_equal(obs, [0, 0])
    np.testing.assert_array_equal(reward, [11, 11])
    np.testing.assert_array_equal(done, [1, 1])
    assert env.task_fingerprint == before


def test_cloned_tasks_with_common_noise_are_paired_identically():
    source = VectorTasks("tabular_mdp", 6, 4, 3, 2, 98)
    left, right = source.clone_slice(1, 5), source.clone_slice(1, 5)
    assert left.task_fingerprints == source.task_fingerprints[1:5]
    assert left.task_fingerprint == right.task_fingerprint
    for t in range(6):
        outputs = [env.step(np.full(4, t % 3, dtype=np.int64)) for env in (left, right)]
        for a, b in zip(outputs[0], outputs[1]):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_paired_evaluation_freezes_weights_isolates_rng_and_counts_episodes(make_system, environment):
    _, policy, _ = make_system(environment)
    policy.train()
    config = {"environment": environment, "num_states": 4, "num_actions": 3,
              "max_episode_len": 2, "meta_episode_len": 6}
    before_weights = {key: value.clone() for key, value in policy.state_dict().items()}
    random.seed(123)
    np.random.seed(321)
    torch.manual_seed(456)
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = evaluate(policy, config, task_seed=40001, action_seed=50001, num_trials=12, batch_size=5)
    assert policy.training
    assert before_python == random.getstate()
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())
    assert all(torch.equal(before_weights[key], value) for key, value in policy.state_dict().items())
    repeat = evaluate(policy, config, task_seed=40001, action_seed=50001, num_trials=12, batch_size=5)
    assert result == repeat
    assert result["paired_tasks_identical"] and result["weights_frozen"]
    episodes = 6 if environment == "bandit" else 3
    for kind in ("policy", "random"):
        per_episode = np.asarray(result[f"{kind}_episode_returns"])
        assert per_episode.shape == (12, episodes)
        np.testing.assert_allclose(per_episode.sum(1), result[kind]["returns"])
        assert result[kind]["ci95_low"] <= result[kind]["mean_return"] <= result[kind]["ci95_high"]
    np.testing.assert_allclose(np.array(result["policy"]["returns"]) - result["random"]["returns"],
                               result["paired_difference"]["returns"])
    other_tasks = evaluate(policy, config, task_seed=40002, action_seed=50001, num_trials=12, batch_size=5)
    assert other_tasks["task_fingerprint"] != result["task_fingerprint"]
