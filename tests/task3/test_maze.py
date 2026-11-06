"""Independent topology, split, pose and real-controller interface checks."""
from collections import Counter
import itertools

import numpy as np
import pytest

from rl2_reproduction.task3.maze_generator import (
    canonical_group_id, cell_center, cell_from_position, enumerate_layouts,
    layout_hash, legal_configurations, path_turn_count, random_kruskal,
    reference_action_plan, shortest_path, split_layouts,
)
from rl2_reproduction.task3.dataset import (
    build_dataset, configuration_key, dataset_capacity, iter_candidate_tasks,
    load_manifest, reference_controller, smoke_validation_subset,
)


def assert_tree(grid):
    cells = {tuple(cell) for cell in np.argwhere(grid == 0)}
    edges = sum((r + 1, c) in cells for r, c in cells) + sum((r, c + 1) in cells for r, c in cells)
    assert edges == len(cells) - 1
    reached = {next(iter(cells))}
    pending = list(reached)
    while pending:
        r, c = pending.pop()
        for neighbour in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if neighbour in cells and neighbour not in reached:
                reached.add(neighbour)
                pending.append(neighbour)
    assert reached == cells


def test_finite_enumeration_and_symmetry():
    layouts = enumerate_layouts()
    assert len(layouts) == len({layout_hash(grid) for grid in layouts}) == 192
    counts = Counter(canonical_group_id(grid) for grid in layouts)
    assert len(counts) == 28
    assert Counter(counts.values()) == {2: 2, 4: 5, 8: 21}
    for grid in layouts:
        assert (grid == 0).sum() == 17
        assert_tree(grid)
        for turns in range(4):
            assert canonical_group_id(np.rot90(grid, turns)) == canonical_group_id(grid)
            assert canonical_group_id(np.fliplr(np.rot90(grid, turns))) == canonical_group_id(grid)


@pytest.mark.parametrize("size,open_cells", [(5, 17), (9, 49)])
def test_kruskal_reproducible_tree(size, open_cells):
    for seed in range(10):
        first = random_kruskal(size, np.random.default_rng(seed))
        second = random_kruskal(size, np.random.default_rng(seed))
        np.testing.assert_array_equal(first, second)
        assert (first == 0).sum() == open_cells
        assert_tree(first)


def test_splits_capacity_and_unique_tasks():
    split = split_layouts(0)
    group_sets = {name: {canonical_group_id(grid) for grid in layouts} for name, layouts in split.items()}
    assert {name: len(groups) for name, groups in group_sets.items()} == {"train": 20, "val": 3, "test": 5}
    assert {name: len(layouts) for name, layouts in split.items()} == {"train": 142, "val": 20, "test": 30}
    for first, second in itertools.combinations(group_sets, 2):
        assert group_sets[first].isdisjoint(group_sets[second])
    capacity = dataset_capacity(0)
    assert [capacity[name]["configuration_capacity"] for name in ("train", "val", "test")] == [108672, 15296, 22784]
    all_keys = set()
    for name, count in (("train", 1000), ("val", 200), ("test", 1000)):
        tasks = list(itertools.islice(iter_candidate_tasks(name), count))
        assert tasks[:10] == list(itertools.islice(iter_candidate_tasks(name), 10))
        for task in tasks:
            key = configuration_key(task)
            assert key not in all_keys
            all_keys.add(key)
            path = shortest_path(task["occupancy"], tuple(task["start_cell"]), tuple(task["goal_cell"]))
            assert len(path) >= 4 and path_turn_count(path) >= 1
            assert task["start_yaw_deg"] in (0, 90, 180, 270)
        assert capacity[name]["configuration_capacity"] > count
        if name == "val":
            subset = smoke_validation_subset(tasks)
            assert len(subset) == 6
            assert set(Counter(item["canonical_group_id"] for item in subset).values()) == {2}


def test_coordinate_roundtrip_and_nominal_plan():
    for row in range(5):
        for col in range(5):
            assert cell_from_position(*cell_center((row, col), 5), 5) == (row, col)
    assert cell_from_position(128, 128, 5) == (3, 1)
    for task in itertools.islice(iter_candidate_tasks("train"), 30):
        actions = reference_action_plan(task)
        assert len(actions) == task["nominal_reference_steps"]
        assert set(actions) <= {0, 1, 2}


class KinematicDebugEnv:
    """Controller interface fixture; never used as evidence for ViZDoom."""
    def reset_episode(self, task):
        self.task = task
        self.pose = [*cell_center(tuple(task["start_cell"]), task["maze_size"]), task["start_yaw_deg"]]
        self.actions = []
        return None, {"pose": self.pose.copy()}

    def step(self, action):
        self.actions.append(action)
        if action in (0, 1):
            self.pose[2] = (self.pose[2] + (5 if action == 0 else -5)) % 360
        else:
            angle = np.deg2rad(self.pose[2])
            self.pose[0] += 25.6 * np.cos(angle)
            self.pose[1] += 25.6 * np.sin(angle)
        success = cell_from_position(*self.pose[:2], self.task["maze_size"]) == tuple(self.task["goal_cell"])
        return None, -0.04 + success, success, {"pose": self.pose.copy(), "success": success}

    def close(self):
        pass


def test_reference_controller_calls_only_legal_actions_and_stops_on_goal():
    env = KinematicDebugEnv()
    tasks = list(itertools.islice(iter_candidate_tasks("val"), 25))
    for task in tasks:
        result = reference_controller(env, task)
        assert result["success"] and result["steps"] <= 200
        assert result["steps"] == len(env.actions)
        assert set(env.actions) <= {0, 1, 2}


def test_reference_controller_budget_is_finite():
    task = next(iter_candidate_tasks("train"))
    result = reference_controller(KinematicDebugEnv(), task, max_steps=1)
    assert not result["success"]
    assert result["steps"] == 1 and result["reason"] == "reference_budget"


def test_manifest_integrity_and_existing_build_protection(tmp_path):
    def scene_fixture(task, path):
        # Explicit interface fixture: this is not claimed to be a WAD engine test.
        path.write_bytes(configuration_key(task).encode())
        return path

    summary = build_dataset(tmp_path, scene_fixture, KinematicDebugEnv, counts={"train": 2, "val": 2, "test": 2})
    assert all(value["task_count"] == 2 for value in summary["splits"].values())
    tasks = load_manifest(tmp_path / "manifests" / "train.jsonl")
    assert len(tasks) == 2 and all(task["reference_controller_steps"] <= 200 for task in tasks)
    with pytest.raises(FileExistsError, match="existing/partial"):
        build_dataset(tmp_path, scene_fixture, KinematicDebugEnv, counts={"train": 2, "val": 2, "test": 2})
    from pathlib import Path
    Path(tasks[0]["scene_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash changed"):
        load_manifest(tmp_path / "manifests" / "train.jsonl")
