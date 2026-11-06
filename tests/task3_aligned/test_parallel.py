import copy
import random

import numpy as np
import pytest
import torch

from rl2_reproduction.task3_aligned.parallel import ParallelCollector, state_dict_digest, trial_spec, wave_size
from rl2_reproduction.task3_aligned.evaluation import aggregate_evaluation, evaluate_parallel
from rl2_reproduction.task3_aligned import parallel


def test_trial_schedule_is_worker_and_resume_independent():
    expected = [trial_spec(0, i, 1000) for i in range(30)]
    assert expected == [trial_spec(0, i, 1000) for i in range(12)] + [trial_spec(0, i, 1000) for i in range(12, 30)]
    assert expected != [trial_spec(1, i, 1000) for i in range(30)]
    assert len({seed for _, seed in expected}) == 30


@pytest.mark.parametrize("target", [1, 499, 500, 501, 1499, 2000, 50000])
def test_wave_schedule_commits_same_trial_prefix_and_respects_budget(target):
    lengths = np.random.default_rng(123).integers(2, 501, 30000)
    counts = []
    for workers in (1, 2, 4):
        count, steps = 0, 0
        while steps < target:
            size = wave_size(target - steps, workers)
            steps += int(lengths[count:count + size].sum())
            count += size
        assert target <= steps <= target + 499
        counts.append((count, steps))
    assert counts[0] == counts[1] == counts[2]


def test_snapshot_owns_cpu_parameters_and_does_not_touch_rng():
    state = {"x": torch.tensor([[1., 2.], [3., 4.]])}
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = copy.deepcopy(np.random.get_state())
    python_rng = random.getstate()
    actors = ParallelCollector({}, workers=4)
    digest = actors.set_weights(state)
    assert digest == state_dict_digest(state)
    state["x"][0, 0] = 99
    assert actors._state_dict["x"][0, 0] == 1
    assert actors.weight_digest == digest
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_rng))
    actors.close()


def _episode(index, length, success):
    return {"episode_index": index, "length": length, "success": success,
            "return": -.04 * length + success, "collision_count": 0,
            "turn_count": length, "turn_fraction": 1.0, "physical_distance": 0.0}


def test_aggregate_keeps_paired_counts_and_empty_success_as_na():
    first = _episode(0, 250, False)
    retain = _episode(1, 250, False)
    reset = _episode(1, 12, True)
    row = {"task_id": "val-one", "episode_summaries": [first, retain],
           "reset_episode_summaries": [first, reset],
           "episode2_return_difference": retain["return"] - reset["return"],
           "episode2_success_difference": -1}
    result = aggregate_evaluation([row], 10001, "paired", 512)
    assert result["policy"]["episodes"][1]["success_count"] == 0
    assert result["reset"]["episodes"][1]["success_count"] == 1
    assert result["paired_episode2"]["both_success_mean_length_improvement_reset_minus_retain"] is None
    assert result["same_episode1_prefix"]


def test_reject_test_training_and_invalid_evaluation_without_starting_workers():
    actors = ParallelCollector({}, workers=1)
    actors.set_weights({"x": torch.ones(1)})
    with pytest.raises(ValueError, match="training tasks"):
        actors.collect([{"split": "test"}], 10)
    with pytest.raises(ValueError, match="unique"):
        evaluate_parallel({}, [{"task_id": "x", "split": "val"}] * 2, {})
    with pytest.raises(ValueError, match="mix"):
        evaluate_parallel({}, [{"task_id": "x", "split": "val"}, {"task_id": "y", "split": "test"}], {})
    actors.close()


def test_worker_rejects_transport_corruption_before_loading(monkeypatch):
    policy = torch.nn.Linear(2, 3)
    original = {key: value.clone() for key, value in policy.state_dict().items()}
    received = {key: torch.ones_like(value) for key, value in original.items()}
    digest = state_dict_digest(received)
    received["weight"][0, 0] = 123
    monkeypatch.setattr(parallel, "_POLICY", policy)
    monkeypatch.setattr(parallel, "_ENV", object())
    monkeypatch.setattr(parallel, "_WEIGHT_DIGEST", None)
    with pytest.raises(RuntimeError, match="transport hash mismatch"):
        parallel._ensure_weights(received, digest)
    assert all(torch.equal(original[key], value) for key, value in policy.state_dict().items())
    received["weight"][0, 0] = 1
    parallel._ensure_weights(received, digest)
    assert state_dict_digest(policy.state_dict()) == digest


def test_actor_failure_propagates_and_cleans_owned_pool(monkeypatch):
    class FailedResult:
        def get(self, timeout):
            raise RuntimeError("engine fixture failure")

    class Worker:
        pid = 1234

    class Pool:
        _pool = [Worker()]

        def apply_async(self, function, arguments):
            return FailedResult()

    actors = ParallelCollector({}, workers=1)
    actors._pool = Pool()
    actors._worker_pids = {1234}
    stopped = []
    monkeypatch.setattr(actors, "_start", lambda: None)
    monkeypatch.setattr(actors, "close", lambda force=False: stopped.append(force))
    with pytest.raises(RuntimeError, match="engine fixture failure") as failure:
        actors._dispatch(None, [{}])
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert stopped == [True]
