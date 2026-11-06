"""Run the unmodified upstream GRU/PPO loop with reproducible small configs."""

import argparse
import contextlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mpi4py
import numpy as np
import torch
from mpi4py import MPI

PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT / "third_party" / "pytorch_rl2"
UPSTREAM_SHA = "51afa4b6c6d05b469c7f74076b9490ae0809b289"


def load_upstream():
    sha = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != UPSTREAM_SHA:
        raise RuntimeError(f"Expected upstream {UPSTREAM_SHA}, found {sha}")
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Upstream has tracked changes; review and record them first.")
    sys.path.insert(0, str(UPSTREAM))
    spec = importlib.util.spec_from_file_location("rl2_reference_train", UPSTREAM / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_net(upstream, config, kind):
    return upstream.create_net(
        net_type=kind, environment=config["environment"], architecture="gru",
        num_states=config["num_states"], num_actions=config["num_actions"],
        num_features=config["num_features"], context_size=config["meta_episode_len"],
    )


def make_optimizer(upstream, net, config):
    return torch.optim.AdamW(
        upstream.get_weight_decay_param_groups(net, config["adam_wd"]),
        lr=config["adam_lr"], eps=config["adam_eps"],
    )


def evaluate(upstream, config, policy, value):
    """Fresh fixed task seeds; preserve training RNG and verify frozen weights."""
    from rl2.algos.common import generate_meta_episode

    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    modes = policy.training, value.training
    before = [p.detach().clone() for net in (policy, value) for p in net.parameters()]
    rows = []
    try:
        policy.eval()
        value.eval()
        for seed in range(config["eval_seed_start"], config["eval_seed_start"] + config["eval_trials"]):
            seed_all(seed)
            env = upstream.create_env(
                config["environment"], config["num_states"],
                config["num_actions"], config["max_episode_len"],
            )
            trial = generate_meta_episode(env, policy, value, config["meta_episode_len"])
            episode_len = env.max_episode_len
            rows.append({
                "seed": seed, "return": float(trial.rews.sum()),
                "episode_returns": [float(x.sum()) for x in trial.rews.reshape(-1, episode_len)],
            })
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        policy.train(modes[0])
        value.train(modes[1])
    after = [p for net in (policy, value) for p in net.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "Evaluation changed weights"
    returns = np.array([r["return"] for r in rows])
    return {"mean_return": float(returns.mean()), "std_return": float(returns.std(ddof=1)), "trials": rows}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(config, output):
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("This validation runner requires one process; MPI scaling is a later task.")
    if config["meta_episode_len"] < 2:
        raise ValueError("The upstream GRU squeezes a length-one sequence; use horizon >= 2.")
    if config["num_states"] <= 0 or config["max_episode_len"] <= 0:
        raise ValueError("num_states and max_episode_len must be positive")
    if config["environment"] == "tabular_mdp" and config["meta_episode_len"] % config["max_episode_len"]:
        raise ValueError("Use complete episodes for the diagnostic evaluation report.")
    for key in ("max_pol_iters", "meta_episodes_per_policy_update", "meta_episodes_per_learner_batch", "ppo_opt_epochs", "num_features", "num_actions"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["eval_trials"] < 2:
        raise ValueError("eval_trials must be at least two")
    upstream = load_upstream()
    from rl2.algos import ppo

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_all(config["seed"])
    env = upstream.create_env(config["environment"], config["num_states"], config["num_actions"], config["max_episode_len"])
    policy, value = [make_net(upstream, config, kind) for kind in ("policy", "value")]
    nets = {"policy": policy, "value": value}
    optimizers = {kind: make_optimizer(upstream, net, config) for kind, net in nets.items()}
    initial = {kind: {k: v.clone() for k, v in net.state_dict().items()} for kind, net in nets.items()}
    before_eval = evaluate(upstream, config, policy, value)
    metrics = []
    original_compute_losses = ppo.compute_losses

    def checked_losses(*args, **kwargs):
        losses = original_compute_losses(*args, **kwargs)
        row = {k: float(v.detach()) for k, v in losses.items()}
        if not all(np.isfinite(x) for x in row.values()):
            raise FloatingPointError(f"Nonfinite PPO loss: {row}")
        metrics.append(row)
        return losses

    def checkpoint(kind, step):
        net = nets[kind]
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            raise FloatingPointError(f"Nonfinite {kind} parameters")
        upstream.save_checkpoint(step, str(output / "checkpoints"), kind, net, optimizers[kind], None)

    start = time.perf_counter()
    ppo.compute_losses = checked_losses
    try:
        ppo.training_loop(
            env=env, policy_net=policy, value_net=value,
            policy_optimizer=optimizers["policy"], value_optimizer=optimizers["value"],
            policy_scheduler=None, value_scheduler=None,
            meta_episodes_per_policy_update=config["meta_episodes_per_policy_update"],
            meta_episodes_per_learner_batch=config["meta_episodes_per_learner_batch"],
            meta_episode_len=config["meta_episode_len"], ppo_opt_epochs=config["ppo_opt_epochs"],
            ppo_clip_param=config["ppo_clip_param"], ppo_ent_coef=config["ppo_ent_coef"],
            discount_gamma=config["discount_gamma"], gae_lambda=config["gae_lambda"],
            standardize_advs=False, max_pol_iters=config["max_pol_iters"], pol_iters_so_far=0,
            policy_checkpoint_fn=lambda step: checkpoint("policy", step),
            value_checkpoint_fn=lambda step: checkpoint("value", step), comm=MPI.COMM_WORLD,
        )
    finally:
        ppo.compute_losses = original_compute_losses
    training_seconds = time.perf_counter() - start
    deltas = {}
    loaded = {}
    for kind, net in nets.items():
        deltas[kind] = float(sum((v - initial[kind][k]).square().sum() for k, v in net.state_dict().items()).sqrt())
        assert deltas[kind] > 0, f"{kind} parameters did not update"
        loaded[kind] = make_net(upstream, config, kind)
        opt = make_optimizer(upstream, loaded[kind], config)
        step = upstream.maybe_load_checkpoint(str(output / "checkpoints"), kind, loaded[kind], opt, None, None)
        assert step == config["max_pol_iters"]
        assert all(torch.equal(v, loaded[kind].state_dict()[k]) for k, v in net.state_dict().items())
    after_eval = evaluate(upstream, config, policy, value)
    assert after_eval == evaluate(upstream, config, loaded["policy"], loaded["value"]), "Checkpoint evaluation differs"
    report = {
        "status": "passed", "purpose": "pipeline smoke validation, not performance reproduction",
        "created_utc": datetime.now(timezone.utc).isoformat(), "upstream_commit": UPSTREAM_SHA,
        "upstream_tracked_changes": False, "config": config,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "mpi4py": mpi4py.__version__,
                    "mpi": MPI.Get_library_version().strip(), "device": "cpu", "torch_threads": 1},
        "training_seconds": training_seconds,
        "training_environment_steps": config["max_pol_iters"] * config["meta_episodes_per_policy_update"] * config["meta_episode_len"],
        "optimizer_steps_per_network": len(metrics), "losses": metrics,
        "parameter_l2_change": deltas, "checkpoint_roundtrip_exact": True,
        "eval_before": before_eval, "eval_after": after_eval,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "training_environment_steps", "training_seconds", "parameter_l2_change")}, indent=2))
    print(f"Full results: {output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output / "train.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            run(config, output)


if __name__ == "__main__":
    main()
