"""Behavioral parity and reproducibility checks for the batched PPO benchmark."""

import copy
import json

import numpy as np
import pytest
import torch

from rl2.algos.common import MetaEpisode, assign_credit
from rl2.algos.ppo import compute_losses as reference_losses
from rl2_reproduction import benchmark
from rl2_reproduction.benchmark_envs import evaluate
from rl2_reproduction.device_adapter import device_adapters


def small_config(environment="tabular_mdp"):
    return {
        "environment": environment, "num_states": 4, "num_actions": 3,
        "num_features": 16, "max_episode_len": 2, "meta_episode_len": 6,
        "trials_per_update": 4, "learner_batch_trials": 2,
        "discount_gamma": 0.99, "gae_lambda": 0.3,
        "ppo_clip_param": 0.1, "ppo_ent_coef": 0.01, "ppo_opt_epochs": 1,
        "adam_lr": 2e-4, "adam_eps": 1e-5, "adam_wd": 0.01,
        "device": "cpu", "torch_threads": 1, "seed": 731,
        "validation_trials": 8, "test_trials": 8, "eval_batch_size": 4,
        "validate_every": 1, "updates": 2,
        "paper_reference_return": 156.2, "paper_random_reference_return": 100.1,
    }


def reference_episodes(data):
    episodes = []
    for i in range(data["obs"].shape[0]):
        meta = MetaEpisode(data["obs"].shape[1], dummy_obs=0)
        for key, values in data.items():
            setattr(meta, key, values[i].detach().cpu().numpy().copy())
        episodes.append(meta)
    return episodes


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_batched_collection_replays_and_matches_reference_gae(make_system, environment):
    _, policy, value = make_system(environment)
    config = small_config(environment)
    data, tasks = benchmark.collect(config, policy, value, task_seed=9341, device="cpu")
    assert len(tasks.task_fingerprints) == config["trials_per_update"]
    assert data["obs"].shape == (4, 6)
    replay = benchmark.replay_check(data, policy, value)
    assert replay["max_logprob_error"] <= 2e-6
    assert replay["max_value_error"] <= 2e-6
    expected_done = np.ones(6) if environment == "bandit" else [0, 1, 0, 1, 0, 1]
    for meta in reference_episodes(data):
        np.testing.assert_array_equal(meta.dones, expected_done)
        # Perturb stored estimates before invoking reference credit assignment,
        # so this asserts the independently recomputed values, not a copied field.
        expected_advantages = meta.advs.copy()
        expected_returns = meta.tdlam_rets.copy()
        meta.advs[:] = 0
        meta.tdlam_rets[:] = 0
        assign_credit(meta, gamma=config["discount_gamma"], lam=config["gae_lambda"])
        np.testing.assert_allclose(meta.advs, expected_advantages, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(meta.tdlam_rets, expected_returns, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
@pytest.mark.parametrize("adapted", [False, True])
def test_tensor_ppo_loss_and_parameter_gradients_match_reference(make_system, environment, adapted):
    _, policy, value = make_system(environment)
    config = small_config(environment)
    data, _ = benchmark.collect(config, policy, value, task_seed=9342, device="cpu")
    # Exercise both sides of the PPO clip with positive and negative advantages.
    data["logpacs"] += torch.tensor([-0.5, 0.5, 0.03, -0.4, 0.4, 0.0]).unsqueeze(0)
    data["advs"] = torch.linspace(-1.7, 1.9, 24).reshape(4, 6)
    new_policy, new_value = copy.deepcopy(policy), copy.deepcopy(value)
    reference = reference_losses(reference_episodes(data), policy, value, clip_param=0.1, ent_coef=0.01)
    reference["policy_loss"].backward()
    reference["value_loss"].backward()
    if adapted:
        with device_adapters():
            actual = benchmark.compute_losses(data, new_policy, new_value, clip_param=0.1, ent_coef=0.01)
            actual["policy_loss"].backward()
            actual["value_loss"].backward()
    else:
        actual = benchmark.compute_losses(data, new_policy, new_value, clip_param=0.1, ent_coef=0.01)
        actual["policy_loss"].backward()
        actual["value_loss"].backward()
    assert set(reference) == set(actual)
    assert actual["clipfrac"].item() > 0
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], atol=2e-6, rtol=2e-5)
    for expected_net, actual_net in ((policy, new_policy), (value, new_value)):
        expected_parameters = dict(expected_net.named_parameters())
        for name, parameter in actual_net.named_parameters():
            expected = expected_parameters[name]
            assert parameter.grad is not None and expected.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected.grad, atol=3e-6, rtol=3e-4)


def test_seed_namespaces_separate_tasks_and_evaluation_preserves_next_rollout(make_system):
    _, policy, value = make_system()
    config = small_config()
    seeds = [benchmark.namespace_seed(config["seed"], namespace, index)
             for namespace in range(5) for index in range(3)]
    assert len(set(seeds)) == len(seeds)
    task_sets = [set(benchmark.make_tasks(config, 8, benchmark.namespace_seed(config["seed"], namespace)).task_fingerprints)
                 for namespace in (0, 1, 2)]
    assert all(not task_sets[i] & task_sets[j] for i in range(3) for j in range(i))
    torch_state = torch.get_rng_state()
    task_seed = benchmark.namespace_seed(config["seed"], 0, 7)
    expected, _ = benchmark.collect(config, policy, value, task_seed, "cpu")
    torch.set_rng_state(torch_state)
    evaluate(policy, config, benchmark.namespace_seed(config["seed"], 1),
             benchmark.namespace_seed(config["seed"], 3), num_trials=8, batch_size=4)
    actual, _ = benchmark.collect(config, policy, value, task_seed, "cpu")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def _assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_checkpoint_resume_restores_rng_optimizer_and_next_training_update(tmp_path):
    config = small_config()
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    uninterrupted.mkdir()
    resumed.mkdir()
    # train() is the real implementation; only CLI paper-size restrictions are
    # bypassed. Each path samples fresh tasks, optimizes, validates, saves RNG,
    # reloads it and evaluates an untouched test namespace.
    with device_adapters():
        benchmark.train(config, uninterrupted)
        partial_config = {**config, "updates": 1}
        benchmark.train(partial_config, resumed)
        benchmark.train(config, resumed, resume=True)
    full = torch.load(uninterrupted / "latest.pt", weights_only=False)
    continued = torch.load(resumed / "latest.pt", weights_only=False)
    assert full["update"] == continued["update"] == 2
    for key in ("policy", "value", "optimizers", "torch_rng", "numpy_rng", "python_rng"):
        _assert_nested_equal(full[key], continued[key])
    full_metrics = [json.loads(line) for line in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    continued_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert len(full_metrics) == len(continued_metrics) == 2
    for left, right in zip(full_metrics, continued_metrics):
        for key in ("update", "training_mean_return", "losses", "training_task_seed", "training_task_batch_fingerprint"):
            assert left[key] == right[key]
