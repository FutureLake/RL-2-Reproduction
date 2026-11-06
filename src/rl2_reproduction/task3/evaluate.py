"""Frozen-weight evaluation with task-keyed RNG and paired hidden-state ablation."""

import hashlib
from pathlib import Path
import random
import numpy as np
import torch
from .trial_collector import TrialCollector, join_episodes
from .metrics import summarize_trials


def _task_id(task):
    value = task.get("task_id") if isinstance(task, dict) else getattr(task, "task_id", None)
    if value is None or not str(value):
        raise ValueError("Evaluation tasks need stable, nonempty task_id values")
    return str(value)


def _generator(seed, task_id, episode):
    digest = hashlib.sha256(str(task_id).encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]
    derived = int(np.random.SeedSequence([seed, *words, episode]).generate_state(1, dtype=np.uint64)[0])
    return torch.Generator(device="cpu").manual_seed(derived)


def select_validation_subset(tasks, per_group=10, expected_groups=3):
    """Fixed first task IDs per sorted validation geometry group; no policy scores."""
    if per_group <= 0:
        raise ValueError("per_group must be positive")
    groups = {}
    seen = set()
    for task in tasks:
        if task.get("split") != "val":
            raise ValueError("The fixed evaluation subset must contain validation tasks only")
        task_id = _task_id(task)
        if task_id in seen:
            raise ValueError("Duplicate validation task_id")
        seen.add(task_id)
        group = task.get("canonical_group_id")
        if not group:
            raise ValueError("Missing validation geometry group")
        groups.setdefault(group, []).append(task)
    if len(groups) != expected_groups or any(len(rows) < per_group for rows in groups.values()):
        raise ValueError(f"Need {expected_groups} validation geometry groups with at least {per_group} tasks each")
    return [task for group in sorted(groups)
            for task in sorted(groups[group], key=_task_id)[:per_group]]


def _save_artifact(directory, task_id, normal, reset_trial=None):
    """Save actual policy input pixels; no environment or network replay."""
    token = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
    path = directory / f"task_{token}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation frames: {path}")
    arrays = {}
    for prefix, trial in (("retain", normal), ("reset", reset_trial)):
        if trial is not None:
            arrays.update({f"{prefix}_{key}": trial.arrays[key]
                           for key in ("images_uint8", "episode_index", "actions", "rewards", "episode_done")})
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fields": sorted(arrays), "images_layout": "uint8 TCHW, RGB, 3x30x40",
            "pixel_source": "actual pre-action observations supplied to policy; excludes terminal diagnostic images"}


@torch.no_grad()
def evaluate_tasks(env, policy, tasks, action_seed=10001, mode="paired", trace_limit=3,
                   trace_task_ids=None, artifact_dir=None, artifact_task_ids=None, progress=None):
    if mode not in ("paired", "rl2", "b1"):
        raise ValueError("Evaluation mode must be paired, rl2 or b1")
    if not tasks:
        raise ValueError("Evaluation needs at least one task")
    identifiers = [_task_id(task) for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evaluation task_id values must be unique")
    if trace_limit is not None and trace_limit < 0:
        raise ValueError("trace_limit must be nonnegative or None")
    trace_task_ids = set(trace_task_ids or ())
    artifact_task_ids = None if artifact_task_ids is None else set(artifact_task_ids)
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    training = policy.training
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    retained, reset, rows = [], [], []
    environment_steps = 0
    collector = TrialCollector(env, policy)
    try:
        policy.eval()
        for index, task in enumerate(tasks):
            task_id = identifiers[index]
            keep_trace = trace_limit is None or index < trace_limit or task_id in trace_task_ids
            first = collector.collect_episode(task, 0, generator=_generator(action_seed, task_id, 0))
            second = collector.collect_episode(task, 1, first.hidden, first.feedback, reset_hidden=(mode == "b1"),
                                               generator=_generator(action_seed, task_id, 1))
            normal = join_episodes([first, second], task_id)
            # Aggregate summaries only. Full-val paired evaluation need not retain
            # hundreds of MB of image arrays after each task has been archived.
            retained.append({"episode_summaries": normal.episode_summaries})
            environment_steps += normal.length
            row = {"task_id": str(task_id), "episode_summaries": normal.episode_summaries,
                   "trial_return": sum(summary["return"] for summary in normal.episode_summaries),
                   "canonical_group_id": task.get("canonical_group_id") if isinstance(task, dict) else getattr(task, "canonical_group_id", None),
                   "reference_controller_steps": task.get("reference_controller_steps") if isinstance(task, dict) else getattr(task, "reference_controller_steps", None)}
            if keep_trace:
                row["trace"] = normal.trace
            reset_trial = None
            if mode == "paired":
                reset_second = collector.collect_episode(task, 1, first.hidden, first.feedback, reset_hidden=True,
                                                         generator=_generator(action_seed, task_id, 1))
                reset_trial = join_episodes([first, reset_second], task_id)
                reset.append({"episode_summaries": reset_trial.episode_summaries})
                environment_steps += reset_second.summary["length"]
                row["reset_episode_summaries"] = reset_trial.episode_summaries
                row["reset_trial_return"] = sum(summary["return"] for summary in reset_trial.episode_summaries)
                row["episode2_return_difference"] = second.summary["return"] - reset_second.summary["return"]
                row["episode2_success_difference"] = int(second.summary["success"]) - int(reset_second.summary["success"])
                if keep_trace:
                    row["reset_trace"] = reset_trial.trace
            if artifact_dir is not None and (artifact_task_ids is None or task_id in artifact_task_ids):
                row["artifact"] = _save_artifact(artifact_dir, task_id, normal, reset_trial)
            rows.append(row)
            if progress is not None:
                progress({"completed": index + 1, "task_count": len(tasks), "task_id": task_id,
                          "environment_steps": environment_steps})
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        policy.train(training)
    if not all(torch.equal(before[name], tensor) for name, tensor in policy.state_dict().items()):
        raise RuntimeError("Evaluation changed model weights")
    result = {"mode": mode, "action_seed": action_seed, "weights_frozen": True, "training_rng_preserved": True,
              "action_rng_scheme": "SeedSequence(action_seed, first four little-endian uint32 words of SHA256(task_id), episode_index)",
              "same_episode1_prefix": mode == "paired", "policy": summarize_trials(retained), "trials": rows,
              "environment_steps": environment_steps, "episodes_executed": len(tasks) * (3 if mode == "paired" else 2)}
    if reset:
        result["reset"] = summarize_trials(reset)
        both_second_success = [row for row in rows if row["episode_summaries"][1]["success"]
                               and row["reset_episode_summaries"][1]["success"]]
        length_improvements = [row["reset_episode_summaries"][1]["length"] - row["episode_summaries"][1]["length"]
                               for row in both_second_success]
        result["paired_episode2"] = {"mean_return_difference": float(np.mean([row["episode2_return_difference"] for row in rows])),
                                     "mean_success_difference": float(np.mean([row["episode2_success_difference"] for row in rows])),
                                     "retain_success_count": result["policy"]["episodes"][1]["success_count"],
                                     "reset_success_count": result["reset"]["episodes"][1]["success_count"],
                                     "retain_timeout_count": result["policy"]["episodes"][1]["timeout_count"],
                                     "reset_timeout_count": result["reset"]["episodes"][1]["timeout_count"],
                                     "both_episode2_success_count": len(both_second_success),
                                     "both_success_mean_length_improvement_reset_minus_retain": float(np.mean(length_improvements)) if length_improvements else None,
                                     "task_count": len(rows), "confidence_interval": None,
                                     "uncertainty_note": "Descriptive paired diagnostic; correlated task configurations and one training seed do not support independent-map or multi-seed inference."}
    return result
