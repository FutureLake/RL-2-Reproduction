"""Real CUDA-graph training must resume the same updates and RNG state."""

import json

import numpy as np
import pytest
import torch

from rl2_reproduction import benchmark
from rl2_reproduction.device_adapter import device_adapters


def _config(environment):
    return {
        "environment": environment, "num_states": 4, "num_actions": 3,
        "num_features": 16, "max_episode_len": 2, "meta_episode_len": 6,
        "trials_per_update": 4, "learner_batch_trials": 2,
        "discount_gamma": 0.99, "gae_lambda": 0.3,
        "ppo_clip_param": 0.1, "ppo_ent_coef": 0.01, "ppo_opt_epochs": 1,
        "adam_lr": 2e-4, "adam_eps": 1e-5, "adam_wd": 0.01,
        "device": "cuda", "torch_threads": 1, "seed": 731,
        "validation_trials": 8, "test_trials": 8, "eval_batch_size": 4,
        "validate_every": 1, "updates": 2, "use_cuda_graph": True,
        "paper_reference_return": 156.2, "paper_random_reference_return": 100.1,
    }


def _assert_exact(expected, actual, path):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0,
                                   msg=lambda message: f"{path}: {message}")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_exact(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(expected, actual)):
            _assert_exact(left, right, f"{path}/{i}")
    else:
        assert actual == expected, path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_cuda_graph_checkpoint_resumes_exact_training_and_rng(environment, tmp_path):
    config = _config(environment)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    uninterrupted.mkdir()
    resumed.mkdir()
    # Exercise the actual CUDA checkpoint load/map_location path, fresh loss
    # graph construction after resume, optimizer restoration and RNG recovery.
    # train() itself disables TF32 and enables deterministic algorithms.
    with device_adapters():
        benchmark.train(config, uninterrupted)
        benchmark.train({**config, "updates": 1}, resumed)
        benchmark.train(config, resumed, resume=True)
    full = torch.load(uninterrupted / "latest.pt", map_location="cpu", weights_only=False)
    continued = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert full["update"] == continued["update"] == 2
    for key in ("policy", "value", "optimizers", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"):
        _assert_exact(full[key], continued[key], key)
    original_metrics = [json.loads(line) for line in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    resumed_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert len(original_metrics) == len(resumed_metrics) == 2
    for original, restored in zip(original_metrics, resumed_metrics):
        for key in ("update", "training_mean_return", "losses", "training_task_seed",
                    "training_task_batch_fingerprint", "validation_mean_return"):
            assert original[key] == restored[key], (key, original[key], restored[key])
    for filename in ("test_final.json", "test_selected.json"):
        original = json.loads((uninterrupted / filename).read_text())
        restored = json.loads((resumed / filename).read_text())
        assert original == restored
