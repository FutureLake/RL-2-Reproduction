"""Budget-only forks preserve training while replacing the final test stream."""

import hashlib
import json

import pytest
import torch

from rl2_reproduction import benchmark
from rl2_reproduction.device_adapter import device_adapters
from test_benchmark_training import small_config, _assert_nested_equal


def file_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fork_and_resume_preserve_training_exactly_and_leave_parent_unchanged(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime required")
    config = {**small_config(), "device": device, "updates": 3,
              "use_cuda_graph": device == "cuda"}
    extended_config = {**config, "test_task_namespace": 12, "test_action_namespace": 14}
    parent, uninterrupted, extension = [tmp_path / name for name in ("parent", "uninterrupted", "extension")]
    for directory in (parent, uninterrupted, extension):
        directory.mkdir()
    with device_adapters():
        benchmark.train({**config, "updates": 1}, parent)
        before = file_hashes(parent)
        benchmark.train(extended_config, uninterrupted)
        benchmark.train({**extended_config, "updates": 2}, extension, resume_from=parent)
        benchmark.train(extended_config, extension, resume=True)
    assert file_hashes(parent) == before
    expected = torch.load(uninterrupted / "latest.pt", map_location="cpu", weights_only=False)
    actual = torch.load(extension / "latest.pt", map_location="cpu", weights_only=False)
    for key in ("policy", "value", "optimizers", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"):
        _assert_nested_equal(expected[key], actual[key])
    for name in ("test_selected.json", "test_final.json"):
        expected_test = json.loads((uninterrupted / name).read_text())
        actual_test = json.loads((extension / name).read_text())
        assert actual_test == expected_test
        assert actual_test["task_seed"] == benchmark.namespace_seed(config["seed"], 12)
        old_test = json.loads((parent / name).read_text())
        assert not set(old_test["task_fingerprints"]) & set(actual_test["task_fingerprints"])
    expected_rows = [json.loads(row) for row in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    actual_rows = [json.loads(row) for row in (extension / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in actual_rows] == [1, 2, 3]
    for left, right in zip(expected_rows, actual_rows):
        for key in ("training_mean_return", "training_task_seed", "training_task_batch_fingerprint",
                    "losses", "validation_mean_return"):
            assert left[key] == right[key]
    manifest = json.loads((extension / "manifest.json").read_text())
    assert manifest["inherited_training_updates_verified"] == 2
    assert manifest["inherited_training_tasks_verified"] == 2 * config["trials_per_update"]
    lineage = manifest["lineage"]
    assert lineage["parent_update"] == 1
    assert lineage["parent_file_sha256"]["latest.pt"] == before["latest.pt"]
    assert lineage["parent_test_namespaces"] == [2, 4]
    assert lineage["extension_test_namespaces"] == [12, 14]
    assert lineage["test_disjointness"]["new_test_vs_validation"]
    assert lineage["test_disjointness"]["new_test_vs_parent_test"]
    assert lineage["test_disjointness"]["new_test_unique_tasks"] == config["test_trials"]


@pytest.fixture
def parent_run(tmp_path):
    config = {**small_config(), "updates": 1}
    parent = tmp_path / "parent"
    parent.mkdir()
    with device_adapters():
        benchmark.train(config, parent)
    return config, parent


@pytest.mark.parametrize("changed", [{"adam_lr": 1e-3}, {"seed": 10}, {"ppo_ent_coef": 0.02},
                                    {"test_task_namespace": 2}, {"test_action_namespace": 4}])
def test_extension_rejects_training_changes_or_reused_test_namespaces(tmp_path, parent_run, changed):
    config, parent = parent_run
    extension = tmp_path / "extension"
    extension.mkdir()
    config = {**config, "updates": 2, "test_task_namespace": 12, "test_action_namespace": 14, **changed}
    before = file_hashes(parent)
    with pytest.raises(ValueError):
        benchmark.prepare_extension(config, extension, parent)
    assert not list(extension.iterdir())
    assert file_hashes(parent) == before


def test_extension_rejects_history_checkpoint_mismatch_before_copy(tmp_path, parent_run):
    config, parent = parent_run
    (parent / "metrics.jsonl").write_text("", encoding="utf-8")
    extension = tmp_path / "extension"
    extension.mkdir()
    with pytest.raises(ValueError, match="history"):
        benchmark.prepare_extension({**config, "updates": 2, "test_task_namespace": 12,
                                     "test_action_namespace": 14}, extension, parent)
    assert not list(extension.iterdir())


def test_history_verification_detects_heldout_overlap_and_seed_tampering(parent_run):
    config, parent = parent_run
    historical = benchmark.make_tasks(config, config["trials_per_update"],
                                      benchmark.namespace_seed(config["seed"], 0, 1))
    with pytest.raises(ValueError, match="held-out"):
        benchmark.validate_training_history(config, parent, 1, set(historical.task_fingerprints))
    rows = [json.loads(row) for row in (parent / "metrics.jsonl").read_text().splitlines()]
    rows[0]["training_task_seed"] += 1
    (parent / "metrics.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        benchmark.validate_training_history(config, parent, 1, set())


@pytest.mark.parametrize("task_namespace,action_namespace", [(0, 14), (1, 14), (3, 14),
                                                           (12, 12), (-1, 14), (True, 14)])
def test_invalid_test_namespaces_are_rejected(task_namespace, action_namespace):
    with pytest.raises(ValueError):
        benchmark.test_namespaces({"test_task_namespace": task_namespace,
                                  "test_action_namespace": action_namespace})
