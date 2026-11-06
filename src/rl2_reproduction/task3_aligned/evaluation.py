"""Parallel frozen-weight evaluation with the audited paired RL² protocol."""
from pathlib import Path
import time

import numpy as np

from ..task3.evaluate import evaluate_tasks
from ..task3.metrics import summarize_trials
from .parallel import ParallelCollector


def _evaluate_one(job):
    from . import parallel
    started = time.perf_counter()
    parallel._ensure_weights(job["state_dict"], job["weight_digest"])
    result = evaluate_tasks(parallel._ENV, parallel._POLICY, [job["task"]],
                            action_seed=job["action_seed"], mode=job["mode"], trace_limit=None,
                            artifact_dir=job["artifact_dir"],
                            artifact_task_ids=[job["task"]["task_id"]] if job["save_frames"] else [])
    return {"row": result["trials"][0], "environment_steps": result["environment_steps"],
            "episodes_executed": result["episodes_executed"],
            "cuda_initialized_after": parallel.torch.cuda.is_initialized(),
            "weight_digest": parallel._WEIGHT_DIGEST, "seconds": time.perf_counter() - started}


def aggregate_evaluation(rows, action_seed, mode, environment_steps):
    retained = [{"episode_summaries": row["episode_summaries"]} for row in rows]
    result = {"mode": mode, "action_seed": action_seed, "weights_frozen": True,
              "training_rng_preserved": True, "same_episode1_prefix": mode == "paired",
              "action_rng_scheme": "SeedSequence(action_seed, first four little-endian uint32 words of SHA256(task_id), episode_index)",
              "policy": summarize_trials(retained), "trials": rows,
              "environment_steps": environment_steps, "episodes_executed": len(rows) * (3 if mode == "paired" else 2)}
    if mode == "paired":
        result["reset"] = summarize_trials([{"episode_summaries": row["reset_episode_summaries"]} for row in rows])
        both = [row for row in rows if row["episode_summaries"][1]["success"]
                and row["reset_episode_summaries"][1]["success"]]
        differences = [row["reset_episode_summaries"][1]["length"] - row["episode_summaries"][1]["length"] for row in both]
        result["paired_episode2"] = {
            "mean_return_difference": float(np.mean([row["episode2_return_difference"] for row in rows])),
            "mean_success_difference": float(np.mean([row["episode2_success_difference"] for row in rows])),
            "retain_success_count": result["policy"]["episodes"][1]["success_count"],
            "reset_success_count": result["reset"]["episodes"][1]["success_count"],
            "retain_timeout_count": result["policy"]["episodes"][1]["timeout_count"],
            "reset_timeout_count": result["reset"]["episodes"][1]["timeout_count"],
            "both_episode2_success_count": len(both),
            "both_success_mean_length_improvement_reset_minus_retain": float(np.mean(differences)) if differences else None,
            "task_count": len(rows), "confidence_interval": None,
            "uncertainty_note": "Descriptive paired diagnostic; related configurations and one training seed are not independent geometry or multi-seed replicates."}
    return result


def evaluate_parallel(policy_state_cpu, tasks, calibration, workers=4, action_seed=10001, mode="rl2",
                      output_artifact_dir=None, progress=None, trace_task_ids=None):
    if mode not in ("rl2", "paired") or not tasks:
        raise ValueError("Evaluation requires nonempty tasks and mode rl2 or paired")
    identifiers = [task["task_id"] for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evaluation task IDs must be unique")
    if any(task.get("split") not in ("val", "test") for task in tasks):
        raise ValueError("Only frozen validation/test tasks may be evaluated here")
    if len({task["split"] for task in tasks}) != 1:
        raise ValueError("Do not mix validation and test rows in one evaluation")
    frame_ids = set(trace_task_ids or ())
    if not frame_ids.issubset(identifiers):
        raise ValueError("Requested frame task is absent from this evaluation")
    if output_artifact_dir is not None:
        output_artifact_dir = str(Path(output_artifact_dir).resolve())
        Path(output_artifact_dir).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, timings = [], []
    total = 0
    with ParallelCollector(calibration, workers=workers, seed=0) as actors:
        digest = actors.set_weights(policy_state_cpu)
        for start in range(0, len(tasks), workers):
            jobs = [{"task": task, "state_dict": actors._state_dict, "weight_digest": digest,
                     "action_seed": action_seed, "mode": mode,
                     "artifact_dir": output_artifact_dir,
                     "save_frames": task["task_id"] in frame_ids}
                    for task in tasks[start:start + workers]]
            completed = actors._dispatch(_evaluate_one, jobs)
            for result in completed:
                if result["weight_digest"] != digest:
                    raise RuntimeError("Evaluation actor returned a different policy version")
                rows.append(result["row"])
                total += result["environment_steps"]
                timings.append({"task_id": result["row"]["task_id"], "seconds": result["seconds"],
                                "cuda_initialized_after": result["cuda_initialized_after"]})
                if progress is not None:
                    progress({"completed": len(rows), "task_count": len(tasks),
                              "task_id": result["row"]["task_id"], "environment_steps": total,
                              "seconds": time.perf_counter() - started})
        if actors.physical_actions != total:
            raise RuntimeError("Evaluation physical action accounting mismatch")
        startup = actors.startup_seconds
        initialization = actors.initialization
    result = aggregate_evaluation(rows, action_seed, mode, total)
    result.update(workers=workers, weight_digest=digest, seconds=time.perf_counter() - started,
                  startup_seconds=startup, task_timings=timings,
                  worker_initialization=initialization,
                  frame_task_ids=sorted(frame_ids), all_task_traces_saved=True,
                  physical_environment_steps=total)
    return result
