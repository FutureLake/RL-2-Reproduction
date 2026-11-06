"""S5 evaluation scheduling, independent RNG and raw-frame archive contracts."""
from collections import Counter
import hashlib
import random

import numpy as np
import pytest
import torch

from rl2_reproduction.task3.evaluate import evaluate_tasks, select_validation_subset
from rl2_reproduction.task3.metrics import summarize_trials
from rl2_reproduction.task3.visual_policy import VisualPolicy


class ShortEnv:
    max_steps_per_episode = 250

    def __init__(self, fail=False):
        self.resets = 0
        self.fail = fail

    def image(self):
        return np.full((3, 30, 40), self.t * 11 + self.task["shade"], dtype=np.uint8)

    def reset_episode(self, task):
        self.task, self.t = task, 0
        self.resets += 1
        return self.image(), {"pose": [64, 64, 0]}

    def step(self, action):
        if self.fail:
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("Deliberate environment failure")
        self.t += 1
        done = self.t == self.task["length"]
        success = done and self.task.get("success", True)
        return None if done else self.image(), -.04 + float(success), done, {
            "pose": [64 + self.t, 64, 0], "success": success, "collided": action == 2, "distance": float(action == 2)}


def policy():
    torch.manual_seed(440)
    return VisualPolicy(12)


def tasks():
    return [{"task_id": "val-a", "length": 3, "shade": 20},
            {"task_id": "val-b", "length": 4, "shade": 90},
            {"task_id": "val-c", "length": 2, "shade": 150, "success": False}]


def assert_rng_equal(before):
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1])
    assert now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_fixed_subset_balances_geometry_and_rejects_test_or_duplicates():
    pool = [{"task_id": f"{g}-{i:02d}", "canonical_group_id": g, "split": "val"}
            for g in ("c", "a", "b") for i in reversed(range(12))]
    chosen = select_validation_subset(pool)
    assert len(chosen) == 30
    assert Counter(t["canonical_group_id"] for t in chosen) == {"a": 10, "b": 10, "c": 10}
    assert [t["task_id"] for t in chosen] == [f"{g}-{i:02d}" for g in ("a", "b", "c") for i in range(10)]
    assert chosen == select_validation_subset(list(reversed(pool)))
    with pytest.raises(ValueError, match="validation tasks only"):
        select_validation_subset([{**pool[0], "split": "test"}, *pool[1:]])
    with pytest.raises(ValueError, match="Duplicate"):
        select_validation_subset(pool + [pool[0]])
    with pytest.raises(ValueError, match="Need 3"):
        select_validation_subset(pool[:12])


def test_task_keyed_draws_identical_in_full_subset_and_reverse_order():
    model = policy()
    full = evaluate_tasks(ShortEnv(), model, tasks(), mode="paired", trace_limit=None)
    backward = evaluate_tasks(ShortEnv(), model, list(reversed(tasks())), mode="paired", trace_limit=None)
    subset = evaluate_tasks(ShortEnv(), model, tasks()[1:2], mode="paired", trace_limit=None)
    by_id = {row["task_id"]: row for row in full["trials"]}
    assert all(row == by_id[row["task_id"]] for row in backward["trials"])
    assert subset["trials"][0] == by_id["val-b"]
    assert full["environment_steps"] == 3 * (3 + 4 + 2)
    assert full["episodes_executed"] == 9
    retain = evaluate_tasks(ShortEnv(), model, tasks(), mode="rl2", trace_limit=None)
    assert retain["environment_steps"] == 2 * (3 + 4 + 2)
    assert "reset" not in retain
    for row in retain["trials"]:
        assert row["trace"] == by_id[row["task_id"]]["trace"]
        assert row["episode_summaries"] == by_id[row["task_id"]]["episode_summaries"]


def test_evaluation_and_progress_restore_all_training_rng_and_mode_even_on_failure():
    model = policy()
    model.train()
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    progress_rows = []
    def progress(row):
        progress_rows.append(row)
        random.random()
        np.random.random()
        torch.rand(())
    result = evaluate_tasks(ShortEnv(), model, tasks(), mode="rl2", progress=progress)
    assert model.training
    assert_rng_equal(before)
    assert progress_rows[-1] == {"completed": 3, "task_count": 3, "task_id": "val-c", "environment_steps": result["environment_steps"]}
    with pytest.raises(RuntimeError, match="Deliberate"):
        evaluate_tasks(ShortEnv(fail=True), model, tasks())
    assert model.training
    assert_rng_equal(before)


def test_actual_frame_artifacts_share_ep1_and_refuse_overwrite(tmp_path):
    model = policy()
    result = evaluate_tasks(ShortEnv(), model, tasks(), artifact_dir=tmp_path,
                            artifact_task_ids={"val-b"}, trace_limit=0, trace_task_ids={"val-b"})
    assert "artifact" not in result["trials"][0] and "trace" not in result["trials"][0]
    row = result["trials"][1]
    artifact = row["artifact"]
    with np.load(artifact["path"], allow_pickle=False) as arrays:
        assert arrays["retain_images_uint8"].dtype == np.uint8
        assert arrays["retain_images_uint8"].shape == (8, 3, 30, 40)
        assert np.array_equal(arrays["retain_images_uint8"][:4], arrays["reset_images_uint8"][:4])
        assert np.array_equal(arrays["retain_actions"][:4], arrays["reset_actions"][:4])
        assert arrays["retain_episode_done"].sum() == 2
        assert np.array_equal(arrays["retain_actions"], [t["action"] for t in row["trace"]])
        assert np.array_equal(arrays["retain_images_uint8"][:, 0, 0, 0], [90, 101, 112, 123] * 2)
    from pathlib import Path
    assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    with pytest.raises(FileExistsError):
        evaluate_tasks(ShortEnv(), model, tasks()[1:2], artifact_dir=tmp_path)


def test_timeout_action_denominators_and_empty_success_values():
    def ep(length, success, turns, collisions):
        return {"length": length, "success": success, "return": -.04 * length + int(success),
                "turn_count": turns, "turn_fraction": turns / length, "collision_count": collisions,
                "physical_distance": 0.0}
    result = summarize_trials([{"episode_summaries": [ep(250, False, 200, 20), ep(100, True, 50, 5)]},
                               {"episode_summaries": [ep(250, False, 100, 25), ep(250, False, 100, 15)]}])
    assert result["episodes"][0]["timeout_count"] == 2
    assert result["episodes"][0]["timeout_rate"] == 1
    assert result["episodes"][0]["mean_success_length"] is None
    assert result["episodes"][0]["success_length_sample_count"] == 0
    assert result["episodes"][1]["success_length_sample_count"] == 1
    assert result["episodes"][1]["mean_success_length"] == 100
    assert result["paired_success_mean_length_improvement"] is None
    assert result["both_success_count"] == 0
    assert result["action_totals"]["total_steps"] == 850
    assert result["action_totals"]["forward_count"] == 400
    assert result["action_totals"]["turn_count"] == 450
    assert result["action_totals"]["collision_count"] == 65
    assert result["action_totals"]["collision_fraction"] == pytest.approx(65 / 850)
    assert result["action_totals"]["collision_per_forward_action"] == pytest.approx(65 / 400)


def test_paired_ep2_empty_success_subset_is_na_and_feedback_unchanged():
    model = policy()
    result = evaluate_tasks(ShortEnv(), model, [tasks()[-1]], trace_limit=None)
    assert result["paired_episode2"]["both_episode2_success_count"] == 0
    assert result["paired_episode2"]["both_success_mean_length_improvement_reset_minus_retain"] is None
    row = result["trials"][0]
    assert row["trace"][:2] == row["reset_trace"][:2]
    assert row["trace"][2]["prev_done"] == row["reset_trace"][2]["prev_done"] == 1
    assert row["trace"][2]["prev_reward"] == row["reset_trace"][2]["prev_reward"]
    assert row["trace"][2]["hidden_norm_before"] > 0
    assert row["reset_trace"][2]["hidden_norm_before"] == 0
