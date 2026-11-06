"""Freeze unique maze tasks only after real-engine reference navigation.

Scene creation and environment construction are injected, so generator tests
do not need to import ViZDoom. No reference actions or poses enter policy data.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .maze_generator import (
    canonical_group_id, cell_center, layout_hash, legal_configurations,
    shortest_path, split_layouts,
)

PROTOCOL = "rl2-task3-from-scratch-v2"
DEFAULT_COUNTS = {"train": 1000, "val": 200, "test": 1000}
SPLIT_NAMESPACES = {"train": 101, "val": 102, "test": 103}


def configuration_key(task: dict) -> str:
    fields = [task["layout_hash"], task["start_cell"], task["goal_cell"], task["start_yaw_deg"]]
    return json.dumps(fields, separators=(",", ":"))


def dataset_capacity(split_seed: int = 0) -> dict[str, dict]:
    result = {}
    for split, layouts in split_layouts(split_seed).items():
        candidates = [legal_configurations(grid) for grid in layouts]
        result[split] = {
            "layout_count": len(layouts),
            "symmetry_group_count": len({canonical_group_id(grid) for grid in layouts}),
            "configuration_capacity": sum(map(len, candidates)),
            "nominal_at_most_200_capacity": sum(item[3] <= 200 for pool in candidates for item in pool),
        }
    return result


def iter_candidate_tasks(split: str, seed: int = 0, split_seed: int = 0) -> Iterator[dict]:
    """Uniform layout draws; legal pose configurations are sampled without replacement.

    The iterator is finite. Exhausted layouts leave the active pool; this event
    is unreachable at normal requested dataset sizes and is recorded by callers.
    """
    if split not in SPLIT_NAMESPACES:
        raise ValueError(f"unknown split: {split}")
    layouts = split_layouts(split_seed)[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_NAMESPACES[split]]))
    pools = [legal_configurations(grid) for grid in layouts]
    permutations = [rng.permutation(len(pool)).tolist() for pool in pools]
    active = list(range(len(layouts)))
    while active:
        index = int(rng.choice(active))
        candidate = pools[index][permutations[index].pop()]
        if not permutations[index]:
            active.remove(index)
        start, goal, yaw, nominal_steps = candidate
        grid = layouts[index]
        task = {
            "protocol_version": PROTOCOL,
            "split": split,
            "maze_size": 5,
            "occupancy": grid.tolist(),
            "layout_hash": layout_hash(grid),
            "canonical_group_id": canonical_group_id(grid),
            "goal_cell": list(goal),
            "start_cell": list(start),
            "start_yaw_deg": yaw,
            "generation_seed": seed,
            "split_seed": split_seed,
            "nominal_reference_steps": nominal_steps,
            "generator_commit": None,
        }
        task["task_id"] = split + "-" + sha256(configuration_key(task).encode()).hexdigest()[:20]
        yield task


def _pose(info: dict) -> tuple[float, float, float]:
    pose = info.get("pose")
    if pose is None or len(pose) != 3:
        raise RuntimeError("reference controller requires privileged info['pose']=(x,y,yaw)")
    result = tuple(map(float, pose))
    if not all(map(math.isfinite, result)):
        raise RuntimeError("nonfinite engine pose")
    return result


def reference_controller(env, task: dict, max_steps: int = 200) -> dict:
    """Closed-loop centerline navigation through the same three env.step actions.

    Pose is privileged diagnostic input only. Small calibrated movement errors
    are corrected by ordinary 5-degree turns, never by teleporting the player.
    """
    _, info = env.reset_episode(task)
    pose = _pose(info)
    path = shortest_path(task["occupancy"], tuple(task["start_cell"]), tuple(task["goal_cell"]))
    waypoints = [cell_center(cell, task["maze_size"]) for cell in path[1:]]
    waypoint = 0
    total_reward = 0.0
    collisions = 0
    distance = 0.0
    for steps in range(1, max_steps + 1):
        x, y, yaw = pose
        # A calibrated forward step is 25.6 units; this tolerance avoids
        # oscillation when tiny accumulated errors place a center behind us.
        while waypoint < len(waypoints) - 1 and math.dist((x, y), waypoints[waypoint]) <= 14.0:
            waypoint += 1
        tx, ty = waypoints[waypoint]
        desired = math.degrees(math.atan2(ty - y, tx - x)) % 360
        delta = (desired - yaw + 180) % 360 - 180
        action = (0 if delta > 0 else 1) if abs(delta) > 2.6 else 2
        _, reward, done, info = env.step(action)
        next_pose = _pose(info)
        distance += math.dist(pose[:2], next_pose[:2])
        pose = next_pose
        total_reward += float(reward)
        collisions += int(bool(info.get("collided", info.get("collision", False))))
        success = bool(info.get("success", info.get("reached_goal", False)))
        if success and not done:
            raise RuntimeError("engine reported success without episode_done")
        if done:
            return {
                "success": success, "steps": steps,
                "reason": "goal" if success else "episode_timeout",
                "return": total_reward, "collisions": collisions,
                "physical_distance": distance,
            }
    return {
        "success": False, "steps": max_steps, "reason": "reference_budget",
        "return": total_reward, "collisions": collisions, "physical_distance": distance,
    }


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_dataset(
    output: str | Path,
    build_scene: Callable,
    env_factory: Callable,
    counts: dict[str, int] | None = None,
    *,
    seed: int = 0,
    split_seed: int = 0,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Build a new frozen dataset; errors propagate instead of becoming failures.

    ``output`` is the protocol root (containing scenes/, manifests/, debug/).
    Existing debug resources are allowed. A partially or fully existing build
    is refused, preserving evidence rather than silently replacing it.
    """
    output = Path(output).resolve()
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("counts must contain positive train/val/test integer counts")
    capacity = dataset_capacity(split_seed)
    for split, count in counts.items():
        if count > capacity[split]["configuration_capacity"]:
            raise ValueError(f"{split}: requested {count} > finite capacity {capacity[split]['configuration_capacity']}")
    manifests = output / "manifests"
    if manifests.exists() and any(manifests.iterdir()):
        raise FileExistsError(f"refusing existing/partial dataset: {manifests}")
    manifests.mkdir(parents=True, exist_ok=True)
    scenes = output / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("maze_generator.py")]
    source_hashes = {path.name: _sha_file(path) for path in source_paths}
    summary = {
        "protocol_version": PROTOCOL, "seed": seed, "split_seed": split_seed,
        "generator_commit": None, "generator_source_sha256": source_hashes,
        "capacity": capacity, "splits": {}, "reference_max_steps": 200,
        "reference_controller": "closed_loop_centerline_same_three_actions",
    }
    all_ids, all_configs, seen_groups = set(), set(), {}
    env = env_factory()
    try:
        for split in ("train", "val", "test"):
            accepted, rejected = [], []
            examined = 0
            consecutive_failures = 0
            for task in iter_candidate_tasks(split, seed, split_seed):
                examined += 1
                task["generator_source_sha256"] = source_hashes
                destination = scenes / split / f"{task['task_id']}.wad"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"refusing existing scene: {destination}")
                scene_path = Path(build_scene(task, destination)).resolve()
                if not scene_path.is_file() or not scene_path.is_relative_to(output):
                    raise RuntimeError("scene builder must return an existing file inside dataset root")
                task["scene_path"] = scene_path.relative_to(output).as_posix()
                task["scene_hash"] = _sha_file(scene_path)
                runtime_task = {**task, "scene_path": str(scene_path)}
                result = reference_controller(env, runtime_task, max_steps=200)
                if not result["success"]:
                    consecutive_failures += 1
                    rejected.append({"task_id": task["task_id"], "result": result, "scene_path": task["scene_path"]})
                    with (manifests / "rejections.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rejected[-1], sort_keys=True) + "\n")
                    if consecutive_failures >= 20:
                        raise RuntimeError("20 consecutive reference-controller failures; inspect calibration and scenes before continuing")
                    continue
                consecutive_failures = 0
                task["reference_controller_steps"] = result["steps"]
                task["reference_controller_result"] = result
                key = configuration_key(task)
                if task["task_id"] in all_ids or key in all_configs:
                    raise AssertionError("duplicate full configuration")
                group = task["canonical_group_id"]
                if group in seen_groups and seen_groups[group] != split:
                    raise AssertionError("geometry group leaked across splits")
                seen_groups[group] = split
                all_ids.add(task["task_id"])
                all_configs.add(key)
                accepted.append(task)
                with (manifests / f"{split}.partial.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n")
                if progress:
                    progress({"split": split, "accepted": len(accepted), "target": counts[split], "examined": examined})
                if len(accepted) == counts[split]:
                    break
            if len(accepted) != counts[split]:
                raise RuntimeError(f"{split}: exhausted finite candidates after accepting {len(accepted)}/{counts[split]}")
            if counts == DEFAULT_COUNTS and len({item["canonical_group_id"] for item in accepted}) != capacity[split]["symmetry_group_count"]:
                raise RuntimeError(f"{split}: frozen dataset would omit a prescribed geometry group")
            (manifests / f"{split}.partial.jsonl").rename(manifests / f"{split}.jsonl")
            summary["splits"][split] = {
                "task_count": len(accepted),
                "layout_count": len({item["layout_hash"] for item in accepted}),
                "symmetry_group_count": len({item["canonical_group_id"] for item in accepted}),
                "candidates_examined": examined, "reference_rejections": len(rejected),
                "reference_rejection_fraction": len(rejected) / examined,
                "rejection_reasons": dict(Counter(item["result"]["reason"] for item in rejected)),
                "start_yaw_counts": dict(Counter(str(item["start_yaw_deg"]) for item in accepted)),
                "manifest_sha256": _sha_file(manifests / f"{split}.jsonl"),
            }
    finally:
        env.close()
    with (output / "dataset_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def load_manifest(path: str | Path, resolve_scenes: bool = True) -> list[dict]:
    path = Path(path).resolve()
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    root = path.parent.parent
    keys, ids = set(), set()
    for task in tasks:
        if task["protocol_version"] != PROTOCOL:
            raise ValueError("manifest protocol version mismatch")
        if layout_hash(task["occupancy"]) != task["layout_hash"] or canonical_group_id(task["occupancy"]) != task["canonical_group_id"]:
            raise ValueError("manifest geometry hash mismatch")
        key = configuration_key(task)
        if task["task_id"] in ids or key in keys:
            raise ValueError("duplicate manifest task")
        ids.add(task["task_id"])
        keys.add(key)
        scene_path = (root / task["scene_path"]).resolve()
        if not scene_path.is_relative_to(root):
            raise ValueError("scene path escapes dataset root")
        if not scene_path.is_file() or _sha_file(scene_path) != task["scene_hash"]:
            raise ValueError(f"scene missing or hash changed: {scene_path}")
        if resolve_scenes:
            task["scene_path"] = str(scene_path)
    return tasks


def smoke_validation_subset(tasks: list[dict]) -> list[dict]:
    """Two sorted configurations per validation geometry group, normally six."""
    if any(task["split"] != "val" for task in tasks):
        raise ValueError("smoke selection accepts validation tasks only")
    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["canonical_group_id"], []).append(task)
    return [task for group in sorted(groups) for task in sorted(groups[group], key=lambda item: item["task_id"])[:2]]
