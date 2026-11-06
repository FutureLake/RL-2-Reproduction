"""Bounded, independently restartable WeightNorm PPO run with frozen provenance."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from rl2_reproduction.task3.dataset import load_manifest
from rl2_reproduction.task3.evaluate import select_validation_subset
from rl2_reproduction.task3.metrics import summarize_trials
from rl2_reproduction.task3.pilot_protocol import model_digest
from rl2_reproduction.task3.sequence_batch import TrialRecord
from rl2_reproduction.task3.train import archive_trials
from rl2_reproduction.task3.utils import atomic_json, configure, namespace_seed, sha256, capture_rng, restore_rng
from .config import validate_config
from .checkpoint import save_checkpoint, load_checkpoint


def source_hashes():
    result = {}
    for directory in (Path(__file__).parent, Path(__file__).parent.parent / "task3"):
        result.update({directory.name + "/" + p.name: sha256(p) for p in sorted(directory.glob("*.py"))})
    return result


def data_identity(config):
    data = Path(config["data_root"]).resolve()
    summary = json.loads((data / "dataset_summary.json").read_text())
    visual = json.loads((data / "visual_config.json").read_text())
    if summary["visual_version"] != config["visual_version"] or visual["visual_version"] != config["visual_version"]:
        raise ValueError("Visual identity mismatch")
    manifest_hashes = {split: sha256(data / f"manifests/{split}.jsonl") for split in ("train", "val", "test")}
    for split in ("train", "val"):
        if manifest_hashes[split] != config["manifest_hashes"][split]:
            raise ValueError(f"Frozen {split} manifest differs")
    if any(summary["manifest_sha256"][s] != h for s, h in manifest_hashes.items()):
        raise ValueError("Dataset summary manifest hashes differ")
    return {"data_root": str(data), "visual_version": config["visual_version"],
            "manifests": manifest_hashes, "visual_config": sha256(data / "visual_config.json"),
            "calibration": sha256(data / "debug/calibration.json"), "sources": source_hashes()}


def _cpu_weights(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def _fixed_examples(tasks, per_group=1):
    groups = sorted({t["canonical_group_id"] for t in tasks})
    return [t["task_id"] for group in groups for t in sorted(
        [t for t in tasks if t["canonical_group_id"] == group], key=lambda t: t["task_id"])[:per_group]]


def _copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def _read_archive(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["task_ids"].tolist()
        for i, task_id in enumerate(ids):
            prefix = f"trial_{i:02d}__"
            yield TrialRecord({k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}, task_id, [])


def _resume_checkpoint(output):
    """Immutable complete-update files are commits; latest is only a pointer."""
    checkpoints = sorted((output / "checkpoints").glob("update_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    if (output / "initial.pt").exists():
        return output / "initial.pt"
    raise FileNotFoundError("Explicit resume needs initial.pt or a complete-update checkpoint")


def run_experiment(config, output, resume=False):
    """No automatic resume, budget extension, or selection on test performance."""
    validate_config(config)
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    if resume:
        _resume_checkpoint(output)
    output.mkdir(parents=True, exist_ok=True)
    # An abandoned lock is deliberately not auto-deleted: verify process exit before resuming.
    lock = output / "active.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        return _run(config, output, resume)
    except BaseException as exc:
        stamp = time.time_ns()
        fault = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                 "time_unix": time.time(), "automatic_restart": False}
        atomic_json(output / f"failure_{stamp}.json", fault)
        atomic_json(output / "status.json", {"state": "failed", **fault})
        raise
    finally:
        lock.unlink(missing_ok=True)


def _run(config, output, resume):
    from .model import WeightNormVisualPolicy
    from .learner import AlignedPPOLearner
    from .parallel import ParallelCollector
    from .evaluation import evaluate_parallel
    from .reporting import generate_report

    run, ppo, evaluation = config["run"], config["ppo"], config["evaluation"]
    configure(run["seed"], threads=run["learner_torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("Approved run requires a working CUDA device")
    hashes = data_identity(config)
    data = Path(config["data_root"])
    calibration = str(data / "debug/calibration.json")
    tasks = load_manifest(data / "manifests/train.jsonl")
    val_tasks = load_manifest(data / "manifests/val.jsonl")
    if len(tasks) != 1000 or len(val_tasks) != 200:
        raise ValueError("Frozen dataset counts changed")
    interim_tasks = select_validation_subset(val_tasks, per_group=evaluation["subset_per_geometry_group"])
    policy = WeightNormVisualPolicy(device="cpu")
    initialization = model_digest(policy)
    policy.to(run["device"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo["learning_rate"], eps=ppo["adam_eps"], weight_decay=ppo["weight_decay"])
    shuffle_rng = torch.Generator(device="cpu").manual_seed(namespace_seed(run["seed"], 40))
    updates, steps, trial_index = 0, 0, 0
    start_time = time.perf_counter()
    if resume:
        committed_checkpoint = _resume_checkpoint(output)
        state = load_checkpoint(committed_checkpoint, policy, optimizer, config=config, hashes=hashes,
                                shuffle_rng=shuffle_rng, restore_random=True)
        _copy_atomic(committed_checkpoint, output / "latest.pt")
        updates, steps, trial_index = state["update"], state["training_steps"], state["next_trial_index"]
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Archived config differs")
        # Complete checkpoints are the commit records; reconcile a crash between checkpoint and metrics.
        committed_rows = []
        for checkpoint in sorted((output / "checkpoints").glob("update_*.pt")):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["update"] <= updates:
                committed_rows.append(saved["extra"]["metrics_row"])
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in committed_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    else:
        atomic_json(output / "config.json", config)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        atomic_json(output / "architecture.json", policy.architecture)
        packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "vizdoom", "Pillow", "matplotlib", "PyYAML", "pytest")}
        atomic_json(output / "manifest.json", {"hashes": hashes, "packages": packages,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "source_root": str(Path(__file__).parent.parent), "initialization": "random, no checkpoint loaded",
                    "model_initial_sha256": initialization, "test_selection": "final checkpoint only"})
        save_checkpoint(output / "initial.pt", policy, optimizer, config, hashes, update=0, training_steps=0,
                        next_trial_index=0, shuffle_rng=shuffle_rng)
        _copy_atomic(output / "initial.pt", output / "latest.pt")
        atomic_json(output / "startup.json", {"data": hashes, "architecture": policy.architecture,
                    "device": str(policy.device), "gpu": torch.cuda.get_device_name(), "seed": run["seed"],
                    "workers": run["workers"], "steps_per_update": run["steps_per_update"],
                    "nominal_training_steps": run["max_training_steps"], "absolute_training_action_cap": run["max_training_steps"] + 499,
                    "model_initial_sha256": initialization, "resume": False})
        print(json.dumps({"startup": json.loads((output / "startup.json").read_text())}), flush=True)
    history = json.loads((output / "evaluation_index.json").read_text()) if (output / "evaluation_index.json").exists() else []
    atomic_json(output / "evaluation_tasks.json", {"initial_val": [t["task_id"] for t in val_tasks],
                "intermediate_val": [t["task_id"] for t in interim_tasks], "fixed_val_examples": _fixed_examples(val_tasks),
                "test_rule": "all 1000 frozen test configurations, final checkpoint, paired, no model selection"})

    def status(phase, **more):
        row = {"state": phase, "update": updates, "training_steps": steps, "next_trial_index": trial_index,
               "elapsed_seconds_this_process": time.perf_counter() - start_time, "time_unix": time.time(), **more}
        atomic_json(output / "status.json", row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def evaluate(label, selected, mode, save_frames=False):
        filename = f"{label}.json"

        def register(result):
            entry = {"file": filename, "sha256": sha256(output / filename), "update": updates,
                     "training_steps": steps, "task_count": len(selected), "mode": mode,
                     "environment_steps": result["environment_steps"], "seconds": result["seconds"]}
            existing = [e for e in history if e["file"] == filename]
            if existing and existing != [entry]:
                raise ValueError("Evaluation index differs from committed result")
            if not existing:
                history.append(entry)
                atomic_json(output / "evaluation_index.json", history)

        if (output / filename).exists():
            prior = json.loads((output / filename).read_text())
            if prior.get("checkpoint_update") != updates or prior.get("model_sha256") != model_digest(policy):
                raise ValueError("Existing evaluation identity differs")
            register(prior)
            return prior
        status("evaluation", evaluation=label, evaluation_tasks=len(selected))
        # Explicitly restore parent RNG even if the implementation changes later.
        rng = capture_rng()
        digest_before = model_digest(policy)
        t = time.perf_counter()
        try:
            # Partial frames from an interrupted attempt remain immutable evidence.
            artifacts = output / "evaluation_frames" / label / f"attempt_{time.time_ns()}" if save_frames else None
            result = evaluate_parallel(_cpu_weights(policy), selected, calibration=calibration,
                workers=run["workers"], action_seed=evaluation["action_seed"], mode=mode,
                output_artifact_dir=artifacts,
                trace_task_ids=_fixed_examples(selected),
                progress=lambda p: status("evaluation", evaluation=label, **p))
        finally:
            restore_rng(rng)
        if model_digest(policy) != digest_before:
            raise RuntimeError("Evaluation changed learner weights")
        result.update(seconds=time.perf_counter() - t, checkpoint_update=updates,
                      training_steps=steps, model_sha256=digest_before, split=selected[0]["split"])
        atomic_json(output / filename, result)
        register(result)
        return result

    def due_evaluations(previous_steps):
        for milestone in evaluation["milestone_training_steps"]:
            if previous_steps < milestone <= steps:
                evaluate(f"validation_milestone_{milestone}", val_tasks, evaluation["milestone_mode"], save_frames=True)
        if updates > 0 and updates % evaluation["every_updates"] == 0 and steps < run["max_training_steps"]:
            evaluate(f"validation_update_{updates:04d}", interim_tasks, evaluation["intermediate_mode"])

    if updates == 0:
        evaluate("validation_initial", val_tasks, evaluation["initial_mode"])
    elif resume:
        last_row = state["extra"]["metrics_row"]
        due_evaluations(steps - int(last_row["collection_metadata"]["env_actions"]))
    learner = AlignedPPOLearner(policy, optimizer, config, shuffle_rng, use_cuda_graph=run["cuda_graph"])
    with ParallelCollector(calibration=calibration, workers=run["workers"], seed=run["seed"]) as collector:
        while steps < run["max_training_steps"]:
            t = time.perf_counter()
            target = min(run["steps_per_update"], run["max_training_steps"] - steps)
            archive_path = output / "rollouts" / f"update_{updates + 1:04d}.npz"
            metadata_path = archive_path.with_suffix(".json")
            inflight_path = output / "collection_inflight.json"
            collector.set_weights(_cpu_weights(policy))
            if archive_path.exists() or metadata_path.exists():
                if not (archive_path.exists() and metadata_path.exists()):
                    raise RuntimeError("Incomplete pending archive; inspect evidence before resume")
                metadata = json.loads(metadata_path.read_text())
                if metadata["start_trial_index"] != trial_index or metadata["policy_sha256"] != model_digest(policy):
                    raise RuntimeError("Pending on-policy archive does not match complete checkpoint")
                if sha256(archive_path) != metadata["archive_sha256"]:
                    raise RuntimeError("Pending archive hash changed")
                batch = list(_read_archive(archive_path))
                if (metadata["next_trial_index"] != trial_index + len(batch)
                        or metadata["trial_count"] != len(batch)
                        or len(metadata["episode_summaries"]) != len(batch)
                        or metadata["task_ids"] != [r.task_id for r in batch]):
                    raise RuntimeError("Pending archive Trial metadata differs")
                for record, summaries in zip(batch, metadata["episode_summaries"]):
                    if len(summaries) != 2 or sum(e["length"] for e in summaries) != record.length:
                        raise RuntimeError("Pending archive Episode lengths differ")
                    record.episode_summaries = summaries
                inflight_path.unlink(missing_ok=True)
                status("reusing_pending_collection", sampled_training_steps=steps + sum(x.length for x in batch))
            else:
                if inflight_path.exists():
                    unfinished = json.loads(inflight_path.read_text())
                    if unfinished["update"] > updates:
                        raise RuntimeError("Unarchived collection was interrupted; refusing to repeat physical actions. "
                                           "Inspect collection_inflight.json before choosing recovery/accounting.")
                    inflight_path.unlink()
                collection_before = collector.physical_actions
                inflight = {"update": updates + 1, "start_trial_index": trial_index,
                            "committed_training_actions": steps, "target_actions": target,
                            "policy_sha256": model_digest(policy), "physical_actions": 0,
                            "time_unix": time.time()}
                atomic_json(inflight_path, inflight)

                def collection_progress(progress):
                    inflight.update(physical_actions=collector.physical_actions - collection_before,
                                    last_progress=progress, time_unix=time.time())
                    atomic_json(inflight_path, inflight)
                    status("training_collection", **progress)

                status("training_collection", target_actions=target)
                try:
                    batch, metadata = collector.collect(tasks, target_steps=target, start_trial_index=trial_index,
                        progress=collection_progress)
                except BaseException:
                    inflight.update(physical_actions=collector.physical_actions - collection_before,
                                    failed=True, time_unix=time.time())
                    atomic_json(inflight_path, inflight)
                    raise
                metadata.update(start_trial_index=trial_index, policy_sha256=model_digest(policy),
                                episode_summaries=[r.episode_summaries for r in batch])
                metadata["archive_sha256"] = archive_trials(archive_path, batch)
                atomic_json(metadata_path, metadata)
                inflight_path.unlink(missing_ok=True)
                if updates == 0:
                    atomic_json(output / "trial_example.json", {"task_id": batch[0].task_id,
                                "episodes": batch[0].episode_summaries, "trace": batch[0].trace})
            actions = sum(record.length for record in batch)
            if not target <= actions <= target + 499 or steps + actions > run["max_training_steps"] + 499:
                raise RuntimeError("Complete Trial physical action budget violated")
            if int(metadata["env_actions"]) != actions:
                raise RuntimeError("Sampled and submitted action counts differ")
            collection_seconds = time.perf_counter() - t
            status("ppo_update", sampled_training_steps=steps + actions, complete_trials=len(batch))
            learn_start = time.perf_counter()
            statistics = learner.update(batch)
            learner_seconds = time.perf_counter() - learn_start
            prior_steps = steps
            steps += actions
            updates += 1
            trial_index = int(metadata["next_trial_index"])
            row = {**statistics, "update": updates, "training_steps": steps,
                   "physical_training_actions": steps, "discarded_training_actions": 0,
                   "seconds": time.perf_counter() - t, "collection_seconds": collection_seconds,
                   "learner_seconds": learner_seconds, "rollout": summarize_trials(batch),
                   "task_ids": [record.task_id for record in batch], "collection_metadata": metadata,
                   "trial_summaries": [record.episode_summaries for record in batch],
                   "rollout_archive_sha256": metadata["archive_sha256"]}
            checkpoint_path = output / "checkpoints" / f"update_{updates:04d}.pt"
            if checkpoint_path.exists():
                raise FileExistsError("Cannot replace a committed update checkpoint")
            save_checkpoint(checkpoint_path, policy, optimizer, config, hashes, update=updates,
                training_steps=steps, next_trial_index=trial_index, shuffle_rng=shuffle_rng, extra={"metrics_row": row})
            _copy_atomic(checkpoint_path, output / "latest.pt")
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            status("update_complete", trial_return=row["rollout"]["mean_trial_return"],
                   collection_seconds=collection_seconds, learner_seconds=learner_seconds)
            del batch
            due_evaluations(prior_steps)
    final_val = evaluate("validation_final", val_tasks, evaluation["final_mode"], save_frames=True)
    # The test set is loaded/scored only after the training budget is complete.
    test_tasks = load_manifest(data / "manifests/test.jsonl")
    if len(test_tasks) != evaluation["test_tasks"]:
        raise ValueError("Final test manifest count differs")
    final_test = evaluate("test_final", test_tasks, evaluation["test_mode"], save_frames=True)
    reloaded = WeightNormVisualPolicy(device=run["device"])
    load_checkpoint(output / "latest.pt", reloaded, config=config, hashes=hashes)
    if model_digest(reloaded) != model_digest(policy):
        raise RuntimeError("Final checkpoint roundtrip differs")
    summary = {"completed": True, "training_steps": steps, "physical_training_actions": steps,
               "discarded_training_actions": 0, "updates": updates, "trials": trial_index,
               "nominal_training_steps": run["max_training_steps"], "absolute_training_action_cap": run["max_training_steps"] + 499,
               "evaluation_environment_steps": sum(e["environment_steps"] for e in history),
               "final_validation": final_val["policy"], "final_test": final_test["policy"],
               "checkpoint_model_exact": True, "test_checkpoint": "final", "test_used_for_selection": False,
               "automatic_extension": False, "seed": run["seed"], "device": run["device"],
               "visual_version": config["visual_version"], "protocol": config["protocol"],
               "scientific_conclusion": "Single-seed descriptive experiment; reset-h is a posthoc ablation, not separately trained B1."}
    atomic_json(output / "summary.json", summary)
    generate_report(output, project_root=config["project_root"])
    status("completed", final_test_success=[e["success_rate"] for e in final_test["policy"]["episodes"]])
    return summary
