"""Bounded single-process visual training; no implicit launch of long experiments."""
from pathlib import Path
import json
import time
import traceback

import numpy as np
import torch

from . import PROTOCOL
from .utils import PROJECT, atomic_json, configure, namespace_seed, provenance, sha256, capture_rng, restore_rng
from .checkpoint import save_checkpoint, load_checkpoint
from .visual_policy import VisualPolicy
from .trial_collector import TrialCollector
from .ppo_learner import PPOLearner
from .evaluate import evaluate_tasks
from .metrics import summarize_trials
from .env import MazeEpisodeEnv


def archive_trials(path, trials):
    """Retain original pixels and old-policy fields for independent replay audits."""
    arrays = {f"trial_{i:02d}__{key}": value for i, trial in enumerate(trials) for key, value in trial.arrays.items()}
    arrays["task_ids"] = np.asarray([trial.task_id for trial in trials])
    arrays["lengths"] = np.asarray([trial.length for trial in trials], dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # A crash can leave the next archive before the update checkpoint exists.
        # Verify deterministic recollection rather than replacing evidence.
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or any(not np.array_equal(prior[k], v) for k, v in arrays.items()):
                raise RuntimeError("Existing next-update rollout differs from deterministic recollection")
        return sha256(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return sha256(path)


def validation_subset(tasks, per_group=2):
    groups = sorted(set(task["canonical_group_id"] for task in tasks))
    return [task for group in groups for task in sorted(
        [t for t in tasks if t["canonical_group_id"] == group], key=lambda t: t["task_id"])[:per_group]]


def train(config, output, resume=False):
    """Record a fault without overwriting an old run or claiming partial updates are resumable."""
    output = Path(output)
    permitted = not output.exists() or (resume and config.get("visual_version") != "rl2-task3-visual-stone-goal-v2")
    try:
        return _train(config, output, resume)
    except BaseException as error:
        if permitted and output.exists():
            status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
            atomic_json(output / "failure.json", {"exception": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "status": status,
                "last_complete_update_checkpoint": str(output / "latest.pt") if (output / "latest.pt").exists() else None,
                "initial_checkpoint": str(output / "initial.pt") if (output / "initial.pt").exists() else None,
                "partial_update_is_not_resumable": True, "automatic_retry": False})
        raise


def _train(config, output, resume=False):
    from .dataset import load_manifest
    from .config import validate_config
    validate_config(config)
    output = Path(output)
    if config["protocol"] != PROTOCOL:
        raise ValueError("Unknown protocol")
    run, ppo = config["run"], config["ppo"]
    s5 = config.get("visual_version") == "rl2-task3-visual-stone-goal-v2"
    if s5 and resume:
        raise ValueError("This S5 invocation is explicitly from random initialization; no checkpoint resume")
    if run["profile"] == "main":
        raise ValueError("Main experiments are intentionally not enabled in S0-S4")
    if run["num_envs"] != 1 or config["env"]["episodes_per_trial"] != 2:
        raise ValueError("Only one process and exactly two episodes are implemented")
    if run["profile"] == "smoke" and (run["updates"] != 5 or run["trials_per_update"] != 8 or run["max_training_steps"] != 20000):
        raise ValueError("Smoke budget must be exactly 5 updates x 8 full trials, cap 20000")
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists: {output}; choose a new run or explicit --resume")
    if resume and not (output / "latest.pt").exists():
        raise FileNotFoundError("Explicit resume needs latest.pt")
    output.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    def status(phase, **fields):
        record = {"phase": phase, "elapsed_seconds": time.perf_counter() - wall_start, **fields}
        atomic_json(output / "status.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)
    status("setup", training_steps=0, device=run["device"], seed=run["seed"])
    configure(run["seed"])
    device = torch.device(run["device"])
    policy = VisualPolicy(hidden_dim=config["model"]["recurrent_hidden_dim"], device=device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo["learning_rate"], eps=ppo["adam_eps"], weight_decay=ppo["weight_decay"])
    task_rng = np.random.default_rng(namespace_seed(run["seed"], 20))
    action_rng = torch.Generator(device="cpu").manual_seed(namespace_seed(run["seed"], 21))
    shuffle_rng = torch.Generator(device="cpu").manual_seed(namespace_seed(run["seed"], 22))
    data = (PROJECT / config["data_root"]).resolve()
    train_path, val_path = data / "manifests/train.jsonl", data / "manifests/val.jsonl"
    tasks = load_manifest(train_path)
    val_tasks = load_manifest(val_path)
    if s5:
        val_tasks = sorted(val_tasks, key=lambda task: task["task_id"])
    evaluation_tasks = validation_subset(val_tasks) if run["profile"] == "smoke" else val_tasks
    source = provenance()
    hashes = {"manifest_train": sha256(train_path), "manifest_val": sha256(val_path),
              "dataset_summary": sha256(data / "dataset_summary.json"), "source": source["task3_source_sha256"]}
    for task in tasks + val_tasks:
        if sha256(task["scene_path"]) != task["scene_hash"]:
            raise ValueError(f"Scene changed: {task['task_id']}")
    calibration = data / "debug/calibration.json"
    if not calibration.exists() or not json.loads(calibration.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("Real-engine calibration must pass before training")
    hashes["calibration"] = sha256(calibration)
    if s5:
        from .pilot_protocol import validate_data, preflight, assert_training_budget, model_digest
        from .evaluate import select_validation_subset
        identity = validate_data(config, data, tasks, val_tasks)
        hashes["visual_config"] = identity["visual_config"]
        hashes["visual_version"] = identity["visual_version"]
        interim_tasks = select_validation_subset(val_tasks, per_group=config["evaluation"]["subset_per_geometry_group"])
        if len(interim_tasks) != 30 or len({t["canonical_group_id"] for t in interim_tasks}) != 3:
            raise ValueError("S5 intermediate validation must have exactly 10 tasks from each of three groups")
        startup = {**identity, "architecture": policy.architecture, "device": str(device), "seed": run["seed"],
                   "parameter_count": sum(p.numel() for p in policy.parameters()),
                   "nominal_training_actions": run["max_training_steps"], "absolute_training_action_cap": 500499,
                   "target_actions_per_update": run["steps_per_update"], "initialization": "seeded random; no checkpoint loaded",
                   "evaluation": config["evaluation"], "training_action_rng_seed": namespace_seed(run["seed"], 21),
                   "initial_model_sha256": model_digest(policy)}
        atomic_json(output / "startup.json", startup)
        print(json.dumps({"event": "startup", **startup}, allow_nan=False), flush=True)
        atomic_json(output / "evaluation_tasks.json", {"full_validation": [t["task_id"] for t in val_tasks],
                    "fixed_val30": [t["task_id"] for t in interim_tasks],
                    "fixed_trajectory_examples": [t["task_id"] for t in validation_subset(val_tasks, per_group=1)],
                    "selection": config["evaluation"]["task_selection"]})
        saved_rng = capture_rng()
        try:
            status("preflight_temporary_model", training_steps=0)
            preflight_result = preflight(config, output, tasks, val_tasks)
        finally:
            restore_rng(saved_rng)
        if preflight_result["model_initial_sha256"] != model_digest(policy):
            raise RuntimeError("Preflight and formal seed-random initialization differ")
    learner = PPOLearner(policy, optimizer, ppo, shuffle_rng)
    update, steps = 0, 0
    if resume:
        state = load_checkpoint(output / "latest.pt", policy, optimizer, config, hashes, task_rng, action_rng, shuffle_rng, True)
        update, steps = state["update"], state["training_steps"]
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        cumulative = 0
        valid_history = True
        for index, row in enumerate(rows, 1):
            cumulative += sum(episode["length"] for trial in row.get("trial_summaries", []) for episode in trial)
            valid_history &= row["update"] == index and row["training_steps"] == cumulative
        if len(rows) != update or not valid_history or (rows and rows[-1]["training_steps"] != steps):
            raise ValueError("Checkpoint/metrics history mismatch")
    else:
        import shutil
        import yaml
        atomic_json(output / "config.json", config)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        atomic_json(output / "architecture.json", policy.architecture)
        atomic_json(output / "manifest.json", {**source, "hashes": hashes, "device": str(device),
                    "initialization": "random; no pretrained or Task1/2 weights", "validation_subset": [t["task_id"] for t in evaluation_tasks]})
        torch.save(policy.state_dict(), output / "initial_weights.pt")
        if s5:
            snapshot = output / "source_snapshot"
            snapshot.mkdir()
            for path in Path(__file__).parent.glob("*.py"):
                shutil.copyfile(path, snapshot / path.name)
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    eval_seed = namespace_seed(run["seed"], 30)
    eval_history = []
    with MazeEpisodeEnv(max_steps=config["env"]["max_steps_per_episode"], calibration=calibration) as env:
        def run_evaluation(label, subset, mode):
            phase = "evaluation_" + label
            status(phase, update=update, training_steps=steps, evaluation_tasks=len(subset), mode=mode)
            saved_task_rng = json.dumps(task_rng.bit_generator.state, sort_keys=True)
            saved_action_rng, saved_shuffle_rng = action_rng.get_state().clone(), shuffle_rng.get_state().clone()
            eval_start = time.perf_counter()
            def progress(row):
                if row["completed"] % 10 == 0 or row["completed"] == row["task_count"]:
                    status(phase, update=update, training_steps=steps, **row)
            kwargs = dict(action_seed=eval_seed, mode=mode)
            if s5:
                kwargs.update(trace_limit=None, progress=progress)
                if label == "final":
                    kwargs["artifact_dir"] = output / "evaluation_frames/final"
            result = evaluate_tasks(env, policy, subset, **kwargs)
            isolated = (saved_task_rng == json.dumps(task_rng.bit_generator.state, sort_keys=True) and
                        torch.equal(saved_action_rng, action_rng.get_state()) and torch.equal(saved_shuffle_rng, shuffle_rng.get_state()))
            if not isolated:
                raise RuntimeError("Evaluation changed a training sampling RNG stream")
            result.update(update=update, training_steps=steps, subset="val200" if len(subset) == 200 else "val30" if len(subset) == 30 else "val_smoke_6",
                          training_sampling_rng_preserved=isolated, seconds=time.perf_counter() - eval_start)
            filename = f"validation_{label}.json"
            atomic_json(output / filename, result)
            environment_steps = sum(sum(e["length"] for e in row["episode_summaries"]) +
                                    (row["reset_episode_summaries"][1]["length"] if "reset_episode_summaries" in row else 0)
                                    for row in result["trials"])
            eval_history.append({"file": filename, "sha256": sha256(output / filename), "update": update,
                                 "training_steps": steps, "task_count": len(subset), "mode": mode,
                                 "environment_steps": environment_steps, "seconds": result["seconds"]})
            atomic_json(output / "evaluation_index.json", eval_history)
            return result
        if not resume:
            save_checkpoint(output / "initial.pt", policy, optimizer, config, hashes, 0, 0, task_rng, action_rng, shuffle_rng)
            evaluation = run_evaluation("initial", evaluation_tasks, config["evaluation"]["initial_mode"] if s5 else "paired" if run.get("mode", "rl2") == "rl2" else "b1")
        collector = TrialCollector(env, policy, action_rng)
        while steps < run["max_training_steps"] and (run["profile"] != "smoke" or update < run["updates"]):
            start = time.perf_counter()
            batch = []
            collected = 0
            target = min(run.get("steps_per_update", 16000), run["max_training_steps"] - steps)
            while True:
                task = tasks[int(task_rng.integers(len(tasks)))]
                if s5 and len(batch) % 5 == 0:
                    status("training_collection", next_update=update + 1, training_steps=steps,
                           sampled_training_steps=steps + collected, completed_trials_in_batch=len(batch), current_task=task["task_id"])
                trial = collector.collect(task, mode=run.get("mode", "rl2"))
                if len(trial.episode_summaries) != 2 or int(trial.arrays["episode_done"].sum()) != 2 or int(trial.arrays["trial_done"].sum()) != 1:
                    raise RuntimeError("Collector violated exactly-two-episode trial protocol")
                batch.append(trial)
                collected = sum(t.length for t in batch)
                if (run["profile"] == "smoke" and len(batch) == run["trials_per_update"]) or (run["profile"] != "smoke" and collected >= target):
                    break
            if run["profile"] == "smoke" and steps + collected > 20000:
                raise RuntimeError("Smoke exceeded training cap")
            if s5:
                assert_training_budget(steps + collected, config)
                status("ppo_update", next_update=update + 1, training_steps=steps,
                       sampled_training_steps=steps + collected, complete_trials=len(batch))
            archive_hash = archive_trials(output / "rollouts" / f"update_{update + 1:04d}.npz", batch)
            statistics = learner.update(batch)
            if not all(statistics["gradients_nonzero"].values()) or statistics["cnn_parameter_l2_change"] <= 0 or statistics["gru_parameter_l2_change"] <= 0:
                raise RuntimeError("Visual/recurrent gradient or actual parameter-change verification failed")
            steps += collected
            update += 1
            row = {**statistics, "update": update, "training_steps": steps, "seconds": time.perf_counter() - start,
                   "rollout": summarize_trials(batch), "task_ids": [trial.task_id for trial in batch],
                   "rollout_archive_sha256": archive_hash,
                   "trial_summaries": [trial.episode_summaries for trial in batch]}
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if update == 1:
                atomic_json(output / "trial_example.json", {"task_id": batch[0].task_id, "episodes": batch[0].episode_summaries, "trace": batch[0].trace})
            save_checkpoint(output / "latest.pt", policy, optimizer, config, hashes, update, steps, task_rng, action_rng, shuffle_rng)
            print(json.dumps({"update": update, "training_steps": steps, "return": row["rollout"]["mean_trial_return"],
                              "loss": row["loss"], "seconds": row["seconds"]}), flush=True)
            if s5 and update % config["evaluation"]["every_updates"] == 0 and steps < run["max_training_steps"]:
                run_evaluation(f"update_{update:04d}", interim_tasks, config["evaluation"]["intermediate_mode"])
        final = run_evaluation("final", evaluation_tasks, config["evaluation"]["final_mode"] if s5 else "paired" if run.get("mode", "rl2") == "rl2" else "b1")
    loaded = VisualPolicy(hidden_dim=config["model"]["recurrent_hidden_dim"], device=device)
    loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=ppo["learning_rate"], eps=ppo["adam_eps"], weight_decay=ppo["weight_decay"])
    load_checkpoint(output / "latest.pt", loaded, loaded_optimizer, config, hashes)
    checkpoint_match = all(torch.equal(t, loaded.state_dict()[k]) for k, t in policy.state_dict().items())
    if not checkpoint_match:
        raise RuntimeError("Checkpoint model roundtrip mismatch")
    summary = {"completed": True, "profile": run["profile"], "updates": update, "training_steps": steps,
               "checkpoint_model_exact": checkpoint_match, "optimizer_state_saved": bool(loaded_optimizer.state_dict()["state"]),
               "parameter_l2_change_this_process": float(sum((t - initial[k]).square().sum() for k, t in policy.state_dict().items()).sqrt()),
               "final_validation": final["policy"], "scientific_conclusion": "Single-seed pilot; interpret measured validation/paired diagnostics descriptively." if s5 else "Smoke verifies engineering only; cross-episode benefit remains unproven."}
    def eval_steps(result):
        return sum(sum(e["length"] for e in row["episode_summaries"]) +
                   (row["reset_episode_summaries"][1]["length"] if "reset_episode_summaries" in row else 0)
                   for row in result["trials"])
    initial_eval = json.loads((output / "validation_initial.json").read_text(encoding="utf-8"))
    summary["evaluation_environment_steps"] = sum(row["environment_steps"] for row in eval_history) if s5 else eval_steps(initial_eval) + eval_steps(final)
    if s5:
        assert_training_budget(steps, config)
        if steps < run["max_training_steps"]:
            raise RuntimeError("Pilot stopped before its requested nominal budget")
        summary.update(visual_version=identity["visual_version"], seed=run["seed"], device=str(device),
                       nominal_training_steps=500000, absolute_training_step_cap=500499,
                       diagnostic_environment_steps=preflight_result["diagnostic_environment_actions"],
                       initialization_matches_temporary_preflight=True, automatic_extension=False,
                       evaluation_schedule=eval_history, elapsed_seconds=time.perf_counter() - wall_start)
    atomic_json(output / "summary.json", summary)
    status("completed", update=update, training_steps=steps, evaluation_environment_steps=summary["evaluation_environment_steps"])
    return summary
