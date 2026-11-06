"""Paper-task-aligned PPO RL2 benchmark with batched sampling and held-out tests."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from rl2_reproduction.baseline import load_upstream, make_net, make_optimizer, seed_all, UPSTREAM_SHA
from rl2_reproduction.benchmark_envs import VectorTasks, evaluate
from rl2_reproduction.device_adapter import device_adapters
from rl2_reproduction.cuda_loss_graph import CudaLossGraph


def namespace_seed(seed, namespace, index=0):
    return int(np.random.SeedSequence([seed, namespace, index]).generate_state(1, dtype=np.uint64)[0])


def make_tasks(config, count, seed):
    return VectorTasks(config["environment"], count, config["num_states"],
                       config["num_actions"], config["max_episode_len"], seed)


@torch.no_grad()
def collect(config, policy, value, task_seed, device):
    batch, horizon = config["trials_per_update"], config["meta_episode_len"]
    env = make_tasks(config, batch, task_seed)
    arrays = {k: torch.empty((batch, horizon), device=device, dtype=dtype)
              for k, dtype in (("obs", torch.long), ("acs", torch.long),
                               ("rews", torch.float32), ("dones", torch.float32),
                               ("logpacs", torch.float32), ("vpreds", torch.float32))}
    observations = env.reset()
    previous_action = torch.zeros(batch, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch, device=device)
    previous_done = torch.ones(batch, device=device)
    policy_state, value_state = policy.initial_state(batch), value.initial_state(batch)
    for t in range(horizon):
        obs = torch.as_tensor(observations, device=device)
        inputs = dict(curr_obs=obs, prev_action=previous_action,
                      prev_reward=previous_reward, prev_done=previous_done)
        distribution, policy_state = policy(**inputs, prev_state=policy_state)
        vpred, value_state = value(**inputs, prev_state=value_state)
        actions = distribution.sample()
        observations, reward, done = env.step(actions.cpu().numpy())
        previous_action = actions
        previous_reward = torch.as_tensor(reward, device=device)
        previous_done = torch.as_tensor(done, device=device)
        for key, tensor in (("obs", obs), ("acs", actions), ("rews", previous_reward),
                            ("dones", previous_done), ("logpacs", distribution.log_prob(actions)),
                            ("vpreds", vpred)):
            arrays[key][:, t] = tensor
    advantages = torch.zeros_like(arrays["rews"])
    future_advantage = torch.zeros(batch, device=device)
    future_value = torch.zeros(batch, device=device)
    for t in reversed(range(horizon)):
        delta = arrays["rews"][:, t] + config["discount_gamma"] * future_value - arrays["vpreds"][:, t]
        future_advantage = delta + config["discount_gamma"] * config["gae_lambda"] * future_advantage
        advantages[:, t] = future_advantage
        future_value = arrays["vpreds"][:, t]
    arrays["advs"] = advantages
    arrays["tdlam_rets"] = arrays["vpreds"] + advantages
    return arrays, env


def sequence_inputs(data):
    batch = data["acs"].shape[0]
    device = data["acs"].device
    return dict(curr_obs=data["obs"],
                prev_action=torch.cat((torch.zeros((batch, 1), dtype=torch.long, device=device), data["acs"][:, :-1]), 1),
                prev_reward=torch.cat((torch.zeros((batch, 1), device=device), data["rews"][:, :-1]), 1),
                prev_done=torch.cat((torch.ones((batch, 1), device=device), data["dones"][:, :-1]), 1))


def compute_losses(data, policy, value, clip_param, ent_coef):
    inputs = sequence_inputs(data)
    batch = data["acs"].shape[0]
    distribution, _ = policy(**inputs, prev_state=policy.initial_state(batch))
    values, _ = value(**inputs, prev_state=value.initial_state(batch))
    entropy = distribution.entropy().mean()
    ratios = (distribution.log_prob(data["acs"]) - data["logpacs"]).exp()
    surrogate = data["advs"] * ratios
    clipped = data["advs"] * ratios.clamp(1 - clip_param, 1 + clip_param)
    error = (data["tdlam_rets"] - values).abs()
    value_loss = torch.where(error < 1, 0.5 * error.square(), error - 0.5).mean()
    return {"policy_loss": -(torch.minimum(surrogate, clipped).mean() + ent_coef * entropy),
            "value_loss": value_loss, "meanent": entropy,
            "clipfrac": (surrogate > clipped).float().mean()}


@torch.no_grad()
def replay_check(data, policy, value):
    inputs = sequence_inputs(data)
    batch = data["acs"].shape[0]
    distribution, _ = policy(**inputs, prev_state=policy.initial_state(batch))
    values, _ = value(**inputs, prev_state=value.initial_state(batch))
    lp_error = (distribution.log_prob(data["acs"]) - data["logpacs"]).abs().max().item()
    value_error = (values - data["vpreds"]).abs().max().item()
    if max(lp_error, value_error) > 1e-4:
        raise RuntimeError(f"Rollout/replay mismatch: {lp_error}, {value_error}")
    return {"max_logprob_error": lp_error, "max_value_error": value_error}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, policy, value, optimizers, update, best_score, config):
    state = {"policy": policy.state_dict(), "value": value.state_dict(),
             "optimizers": [o.state_dict() for o in optimizers], "update": update,
             "best_validation_score": best_score, "config": config,
             "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(policy.parameters()).is_cuda else None}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def test_namespaces(config):
    """Keep evaluation streams separate from training and fixed validation."""
    namespaces = (config.get("test_task_namespace", 2), config.get("test_action_namespace", 4))
    if any(type(n) is not int or n < 0 or n in (0, 1, 3) for n in namespaces):
        raise ValueError("Test namespaces must be nonnegative integers outside training/validation namespaces 0, 1, 3")
    if namespaces[0] == namespaces[1]:
        raise ValueError("Test task and action namespaces must differ")
    return namespaces


def validate_resume_config(config, prior, allow_new_test=False):
    current, previous = dict(config), dict(prior)
    for settings in (current, previous):
        settings.setdefault("test_task_namespace", 2)
        settings.setdefault("test_action_namespace", 4)
    allowed = {"updates"}
    if allow_new_test:
        allowed.update(("test_task_namespace", "test_action_namespace"))
    changed = {key for key in current.keys() | previous.keys()
               if key not in allowed and current.get(key) != previous.get(key)}
    if changed:
        raise ValueError(f"Resume configuration differs in fixed settings: {sorted(changed)}")


def validate_training_history(config, source, checkpoint_update, held_out):
    """Regenerate prior tasks, verifying history and new-test disjointness.

    This uses local NumPy Generators only and does not consume training RNG.
    The returned fingerprints also prevent duplicates across a resume boundary.
    """
    rows = [json.loads(line) for line in (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["update"] for row in rows] != list(range(1, checkpoint_update + 1)):
        raise ValueError("Training history must contain updates 1..checkpoint_update exactly once")
    fingerprints = set()
    for row in rows:
        expected_seed = namespace_seed(config["seed"], 0, row["update"])
        if row["training_task_seed"] != expected_seed:
            raise ValueError("Historical training seed differs from the declared namespace")
        tasks = make_tasks(config, config["trials_per_update"], expected_seed)
        if row["training_task_batch_fingerprint"] != tasks.task_fingerprint:
            raise ValueError("Historical training task fingerprint could not be reproduced")
        current = set(tasks.task_fingerprints)
        if len(current) != config["trials_per_update"] or current & held_out or current & fingerprints:
            raise ValueError("Duplicate or held-out task found in historical training")
        fingerprints.update(current)
    return fingerprints


def prepare_extension(config, output, source):
    """Fork trusted local results without changing the parent or copying tests."""
    output, source = output.resolve(), Path(source).resolve()
    if source == output or not source.is_dir() or any(output.iterdir()):
        raise ValueError("Extension requires a distinct source and an empty output directory")
    checkpoint = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    validate_resume_config(config, checkpoint["config"], allow_new_test=True)
    new_namespaces, old_namespaces = test_namespaces(config), test_namespaces(checkpoint["config"])
    if set(new_namespaces) & set(old_namespaces):
        raise ValueError("An extension must use fresh test task and action namespaces")
    if config["updates"] <= checkpoint["update"]:
        raise ValueError("Extension target updates must exceed the parent checkpoint")
    new_test = make_tasks(config, config["test_trials"], namespace_seed(config["seed"], new_namespaces[0]))
    validation = make_tasks(config, config["validation_trials"], namespace_seed(config["seed"], 1))
    parent_config = checkpoint["config"]
    old_test = make_tasks(parent_config, parent_config["test_trials"],
                          namespace_seed(parent_config["seed"], old_namespaces[0]))
    new_fingerprints = set(new_test.task_fingerprints)
    if new_fingerprints & set(validation.task_fingerprints) or new_fingerprints & set(old_test.task_fingerprints):
        raise ValueError("Extension test tasks overlap validation or parent test tasks")
    if len(new_fingerprints) != config["test_trials"]:
        raise ValueError("Extension test tasks are not unique")
    best = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    if best["update"] > checkpoint["update"] or best["best_validation_score"] != checkpoint["best_validation_score"]:
        raise ValueError("Parent best and latest checkpoint selection metadata disagree")
    filenames = ("latest.pt", "best.pt", "metrics.jsonl", "validation_initial.json", "validation_latest.json")
    source_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                     for name in (*filenames, "manifest.json")}
    lineage = {"parent_directory": str(source), "parent_update": checkpoint["update"],
               "parent_file_sha256": source_hashes,
               "parent_test_namespaces": list(old_namespaces),
               "extension_test_namespaces": list(new_namespaces),
               "test_disjointness": {"new_test_vs_validation": True, "new_test_vs_parent_test": True,
                                     "new_test_unique_tasks": len(new_fingerprints),
                                     "new_test_batch_fingerprint": new_test.task_fingerprint,
                                     "validation_batch_fingerprint": validation.task_fingerprint,
                                     "parent_test_batch_fingerprint": old_test.task_fingerprint},
               "training_and_validation_settings_unchanged": True}
    # Validate before creating any inherited output. train() also checks the
    # reconstructed tasks against the new held-out set before taking a step.
    rows = [json.loads(line) for line in (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["update"] for row in rows] != list(range(1, checkpoint["update"] + 1)):
        raise ValueError("Parent history and checkpoint updates disagree")
    for name in filenames:
        shutil.copy2(source / name, output / name)
    atomic_json(output / "lineage.json", lineage)
    return lineage


def train(config, output, resume=False, resume_from=None):
    if resume and resume_from is not None:
        raise ValueError("resume and resume_from are mutually exclusive")
    test_task_namespace, test_action_namespace = test_namespaces(config)
    lineage = None
    if resume_from is not None:
        lineage = prepare_extension(config, output, resume_from)
    elif resume and (output / "lineage.json").exists():
        lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
        if list((test_task_namespace, test_action_namespace)) != lineage["extension_test_namespaces"]:
            raise ValueError("Resume test namespaces differ from the extension lineage")
    continuing = resume or resume_from is not None
    up = load_upstream()
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config.get("torch_threads", 2))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(config["seed"])
    policy, value = [make_net(up, config, kind).to(device) for kind in ("policy", "value")]
    optimizers = [make_optimizer(up, net, config) for net in (policy, value)]
    val_seed = namespace_seed(config["seed"], 1)
    test_seed = namespace_seed(config["seed"], test_task_namespace)
    held_out = set(make_tasks(config, config["validation_trials"], val_seed).task_fingerprints)
    held_out.update(make_tasks(config, config["test_trials"], test_seed).task_fingerprints)
    training_tasks = set()
    start_update, best_score = 0, -float("inf")
    if continuing:
        # Trusted, locally generated checkpoint contains RNG tuples as well as tensors.
        checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=False)
        prior = checkpoint["config"]
        pending_fork = lineage is not None and checkpoint["update"] == lineage["parent_update"]
        validate_resume_config(config, prior, allow_new_test=pending_fork)
        if config["updates"] < checkpoint["update"]:
            raise ValueError("Target updates cannot precede the loaded checkpoint")
        training_tasks = validate_training_history(config, output, checkpoint["update"], held_out)
        policy.load_state_dict(checkpoint["policy"])
        value.load_state_dict(checkpoint["value"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        start_update, best_score = checkpoint["update"], checkpoint["best_validation_score"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
    metadata = {"upstream_commit": UPSTREAM_SHA, "config": config,
                "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "device": str(device), "hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                "training_seed_namespace": [config["seed"], 0, "update"],
                "validation_task_seed": val_seed, "test_task_seed": test_seed,
                "test_task_namespace": test_task_namespace, "test_action_namespace": test_action_namespace,
                "inherited_training_updates_verified": start_update,
                "inherited_training_tasks_verified": len(training_tasks), "lineage": lineage,
                "selection": "maximum fixed-validation mean; test not used for selection",
                "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("benchmark.py", "benchmark_envs.py", "device_adapter.py", "cuda_loss_graph.py")},
                "implementation": "project batched sampler and device-aware upstream-equivalent PPO; custom upstream GRU"}
    atomic_json(output / "manifest.json", metadata)
    if not continuing:
        initial_validation = evaluate(policy, config, val_seed, namespace_seed(config["seed"], 3),
                                      config["validation_trials"], device, config["eval_batch_size"])
        atomic_json(output / "validation_initial.json", initial_validation)
        best_score = initial_validation["policy"]["mean_return"]
        save_checkpoint(output / "best.pt", policy, value, optimizers, 0, best_score, config)
    start_time = time.perf_counter()
    loss_graph = None
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for update in range(start_update + 1, config["updates"] + 1):
            tick = time.perf_counter()
            data, tasks = collect(config, policy, value, namespace_seed(config["seed"], 0, update), device)
            fingerprints = set(tasks.task_fingerprints)
            if fingerprints & held_out or fingerprints & training_tasks:
                raise RuntimeError("Duplicate or held-out task found in training")
            training_tasks.update(fingerprints)
            replay = replay_check({k: v[:8] for k, v in data.items()}, policy, value) if update == start_update + 1 else None
            records = []
            for epoch in range(config["ppo_opt_epochs"]):
                order = np.random.permutation(config["trials_per_update"])
                for offset in range(0, len(order), config["learner_batch_trials"]):
                    indexes = torch.as_tensor(order[offset:offset + config["learner_batch_trials"]], device=device)
                    minibatch = {k: v[indexes] for k, v in data.items()}
                    if config.get("use_cuda_graph", False):
                        if loss_graph is None:
                            loss_graph = CudaLossGraph(compute_losses, policy, value, minibatch,
                                                      config["ppo_clip_param"], config["ppo_ent_coef"])
                        losses = loss_graph.run(minibatch)
                    else:
                        losses = compute_losses(minibatch, policy, value, config["ppo_clip_param"], config["ppo_ent_coef"])
                    scalars = {key: loss.detach().item() for key, loss in losses.items()}
                    if not all(np.isfinite(x) for x in scalars.values()):
                        raise FloatingPointError(scalars)
                    for name, net, optimizer in zip(("policy_loss", "value_loss"), (policy, value), optimizers):
                        if loss_graph is None:
                            optimizer.zero_grad(set_to_none=True)
                            losses[name].backward()
                        gradients = [p.grad for p in net.parameters()]
                        if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
                            raise FloatingPointError("Nonfinite or missing gradient")
                        optimizer.step()
                    records.append(scalars)
            row = {"update": update, "environment_steps": update * config["trials_per_update"] * config["meta_episode_len"],
                   "training_mean_return": data["rews"].sum(1).mean().item(),
                   "losses": {k: float(np.mean([r[k] for r in records])) for k in records[0]},
                   "optimizer_steps_per_network": len(records), "training_task_seed": tasks.task_seed,
                   "training_task_batch_fingerprint": tasks.task_fingerprint,
                   "no_training_heldout_overlap": True, "replay_check": replay}
            if update % config["validate_every"] == 0 or update == config["updates"]:
                validation = evaluate(policy, config, val_seed, namespace_seed(config["seed"], 3),
                                      config["validation_trials"], device, config["eval_batch_size"])
                row["validation_mean_return"] = validation["policy"]["mean_return"]
                atomic_json(output / "validation_latest.json", validation)
                if row["validation_mean_return"] > best_score:
                    best_score = row["validation_mean_return"]
                    save_checkpoint(output / "best.pt", policy, value, optimizers, update, best_score, config)
            row["seconds"] = time.perf_counter() - tick
            row["elapsed_seconds_this_session"] = time.perf_counter() - start_time
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            save_checkpoint(output / "latest.pt", policy, value, optimizers, update, best_score, config)
            print(f"{config['environment']} update {update}/{config['updates']} steps={row['environment_steps']:,} "
                  f"train={row['training_mean_return']:.2f} val={row.get('validation_mean_return','-')} seconds={row['seconds']:.2f}", flush=True)
    # Selection is fixed before either checkpoint is evaluated on the final test tasks.
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    final_test = evaluate(policy, config, test_seed, namespace_seed(config["seed"], test_action_namespace),
                          config["test_trials"], device, config["eval_batch_size"])
    policy.load_state_dict(selected["policy"])
    selected_test = evaluate(policy, config, test_seed, namespace_seed(config["seed"], test_action_namespace),
                             config["test_trials"], device, config["eval_batch_size"])
    atomic_json(output / "test_final.json", final_test)
    atomic_json(output / "test_selected.json", selected_test)
    target = config["paper_reference_return"]
    result = {"status": "completed", "config": config, "selected_update": selected["update"],
              "selected_validation_return": selected["best_validation_score"],
              "test_policy": selected_test["policy"], "test_random": selected_test["random"],
              "paired_difference": selected_test["paired_difference"],
              "final_checkpoint_test_policy": final_test["policy"],
              "paper_reference_return": target,
              "gap_to_paper": target - selected_test["policy"]["mean_return"],
              "paper_gain_recovered_fraction": (selected_test["policy"]["mean_return"] - selected_test["random"]["mean_return"]) /
                    (target - config["paper_random_reference_return"]),
              "task_uncertainty_only_single_training_seed": True,
              "training_environment_steps": config["updates"] * config["trials_per_update"] * config["meta_episode_len"]}
    atomic_json(output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "test_policy", "test_random", "paired_difference", "final_checkpoint_test_policy")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true")
    continuation.add_argument("--resume-from", type=Path,
                              help="Fork an existing run into a new directory with fresh test namespaces")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config["num_features"] != 256 or config["meta_episode_len"] != 100 or config["num_actions"] != 5:
        raise ValueError("Aligned benchmark requires GRU256, five actions and 100-step trials")
    if config["trials_per_update"] % config["learner_batch_trials"]:
        raise ValueError("Use full equal learner minibatches")
    if config["environment"] == "tabular_mdp" and (config["num_states"] != 10 or config["max_episode_len"] != 10):
        raise ValueError("Paper MDP requires ten states and ten-step episodes")
    if config["environment"] not in ("bandit", "tabular_mdp"):
        raise ValueError("Unsupported benchmark task")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    load_upstream()
    with device_adapters():
        train(config, output, args.resume, args.resume_from)


if __name__ == "__main__":
    main()
