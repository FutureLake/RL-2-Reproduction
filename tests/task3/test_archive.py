"""Persisted rollout evidence is lossless and cannot silently replace a batch."""
import numpy as np
import pytest
from rl2_reproduction.task3.sequence_batch import TrialRecord
from rl2_reproduction.task3.train import archive_trials


def test_rollout_archive_lossless_types_and_recollection(tmp_path):
    rng = np.random.default_rng(19)
    arrays = {"images_uint8": rng.integers(0, 256, (3, 3, 30, 40), dtype=np.uint8),
              "actions": np.array([0, 2, 1], dtype=np.int64), "rewards": np.array([-.04, .96, -.04], dtype=np.float32),
              "episode_done": np.array([False, True, True]), "old_logits": rng.normal(size=(3, 3)).astype(np.float32)}
    trial = TrialRecord(arrays, "explicit-test-fixture", [])
    path = tmp_path / "rollouts/update_0001.npz"
    before = archive_trials(path, [trial])
    with np.load(path, allow_pickle=False) as stored:
        assert stored["task_ids"].tolist() == [trial.task_id]
        assert stored["lengths"].tolist() == [3]
        for key, value in arrays.items():
            assert stored[f"trial_00__{key}"].dtype == value.dtype
            np.testing.assert_array_equal(stored[f"trial_00__{key}"], value)
    assert archive_trials(path, [trial]) == before
    arrays["actions"][0] = 1
    with pytest.raises(RuntimeError, match="differs"):
        archive_trials(path, [trial])
