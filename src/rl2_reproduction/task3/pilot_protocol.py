"""S5 provenance, preflight and budget checks; no policy/environment changes."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .utils import PROJECT, atomic_json, configure, namespace_seed, sha256

VISUAL_VERSION = "rl2-task3-visual-stone-goal-v2"
MANIFEST_HASHES = {
    "train": "b9929e27ef54b48ed6b48a46548fe144be705a551082374c335ec64b95c927b2",
    "val": "c3464e804f2bccbcb12a3b50ca60835765b61576717e81bf0a012b6810b2bbce",
}
EVALUATION = {"initial_tasks": 200, "final_tasks": 200, "every_updates": 5,
              "subset_per_geometry_group": 10, "initial_mode": "rl2", "intermediate_mode": "rl2",
              "final_mode": "paired", "task_selection": "sorted_task_id_first_10_per_validation_geometry_group"}


def model_digest(policy):
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_data(config, data, train_tasks, val_tasks):
    """Pin the accepted corpus, not merely a directory with a similar name."""
    data = Path(data).resolve()
    if config.get("visual_version") != VISUAL_VERSION:
        raise ValueError("S5 requires the accepted stone-goal-v2 visual version")
    if data != (PROJECT / "data/task3" / VISUAL_VERSION).resolve():
        raise ValueError("S5 data_root is not the accepted frozen directory")
    summary = json.loads((data / "dataset_summary.json").read_text())
    visual = json.loads((data / "visual_config.json").read_text())
    if summary["visual_version"] != VISUAL_VERSION or visual["visual_version"] != VISUAL_VERSION:
        raise ValueError("Dataset visual metadata mismatch")
    if len(train_tasks) != 1000 or len(val_tasks) != 200:
        raise ValueError("S5 expects the original 1000 training and 200 validation tasks")
    visual_hash = sha256(data / "visual_config.json")
    for split, tasks in (("train", train_tasks), ("val", val_tasks)):
        actual = sha256(data / f"manifests/{split}.jsonl")
        if actual != MANIFEST_HASHES[split] or summary["manifest_sha256"][split] != actual:
            raise ValueError(f"Frozen {split} manifest hash mismatch")
        if any(t["visual_version"] != VISUAL_VERSION or t["split"] != split or
               t["visual_config_sha256"] != visual_hash for t in tasks):
            raise ValueError(f"Wrong visual version/split/config hash in {split} tasks")
    return {"data_root": str(data), "visual_version": VISUAL_VERSION,
            "manifest_train": MANIFEST_HASHES["train"], "manifest_val": MANIFEST_HASHES["val"],
            "visual_config": visual_hash, "counts": {"train": 1000, "val": 200},
            "test_manifest_loaded_or_scored": False}


def assert_training_budget(steps, config):
    nominal = config["run"]["max_training_steps"]
    maximum_trial = config["env"]["episodes_per_trial"] * config["env"]["max_steps_per_episode"]
    if steps > nominal + maximum_trial - 1:
        raise RuntimeError(f"Training actions {steps} exceed absolute cap {nominal + maximum_trial - 1}")


def preflight(config, output, tasks=None, val_tasks=None):
    """One temporary random model and one real complete trial, never a PPO update."""
    from PIL import Image
    from .config import validate_config
    from .dataset import load_manifest
    from .env import MazeEpisodeEnv, preprocess_rgb
    from .trial_collector import TrialCollector
    from .visual_policy import VisualPolicy
    validate_config(config)
    output = Path(output)
    if (output / "preflight.json").exists():
        raise FileExistsError("Preserve earlier preflight evidence")
    output.mkdir(parents=True, exist_ok=True)
    data = (PROJECT / config["data_root"]).resolve()
    tasks = tasks if tasks is not None else load_manifest(data / "manifests/train.jsonl")
    val_tasks = val_tasks if val_tasks is not None else load_manifest(data / "manifests/val.jsonl")
    identity = validate_data(config, data, tasks, val_tasks)
    configure(config["run"]["seed"])
    policy = VisualPolicy(hidden_dim=config["model"]["recurrent_hidden_dim"], device="cpu")
    initial = model_digest(policy)
    diagnostic_rng = torch.Generator(device="cpu").manual_seed(namespace_seed(config["run"]["seed"], 50))
    task = sorted(tasks, key=lambda t: t["task_id"])[0]
    with MazeEpisodeEnv(calibration=data / "debug/calibration.json") as env:
        frame, _ = env.reset_episode(task)
        assert frame.shape == (3, 30, 40) and frame.dtype == np.uint8
        assert np.array_equal(frame, preprocess_rgb(env.last_rgb))
        Image.fromarray(env.last_rgb).save(output / "preflight_160x120.png")
        Image.fromarray(frame.transpose(1, 2, 0)).save(output / "preflight_40x30.png")
        trial = TrialCollector(env, policy, diagnostic_rng).collect(task)
    a = trial.arrays
    assert len(trial.episode_summaries) == int(a["episode_done"].sum()) == 2
    assert int(a["trial_done"].sum()) == 1 and bool(a["trial_done"][-1])
    boundary = trial.episode_summaries[0]["length"]
    assert a["prev_episode_done"][boundary] == 1 and not a["reset_mask"][boundary]
    assert a["prev_actions"][boundary] == a["actions"][boundary - 1]
    assert a["prev_rewards"][boundary] == a["rewards"][boundary - 1]
    assert trial.episode_summaries[0]["start_pose"] == trial.episode_summaries[1]["start_pose"]
    assert trial.trace[boundary]["hidden_norm_before"] == trial.trace[boundary - 1]["hidden_norm_after"]
    assert model_digest(policy) == initial
    result = {"passed": True, "identity": identity, "task_id": task["task_id"], "seed": config["run"]["seed"],
              "temporary_model": True, "model_initial_sha256": initial, "ppo_updates": 0,
              "diagnostic_environment_actions": trial.length, "excluded_from_training_budget": True,
              "exactly_two_complete_episodes": True, "episode2_feedback_and_hidden_retained": True,
              "observation_shape_chw": list(a["images_uint8"].shape[1:]), "episodes": trial.episode_summaries,
              "trace": trial.trace}
    atomic_json(output / "preflight.json", result)
    return result
