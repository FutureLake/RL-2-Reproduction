"""Atomic update-boundary checkpoints including all sampling and optimizer state."""
from pathlib import Path
import torch
from .utils import capture_rng, restore_rng


def save_checkpoint(path, policy, optimizer, config, hashes, update, steps, task_rng, action_rng, shuffle_rng):
    state = {"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config, "hashes": hashes, "update": update, "training_steps": steps,
             "rng": capture_rng(), "task_rng": task_rng.bit_generator.state,
             "action_rng": action_rng.get_state(), "shuffle_rng": shuffle_rng.get_state(),
             "architecture": policy.architecture, "boundary": "complete_update"}
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, policy, optimizer=None, config=None, hashes=None,
                    task_rng=None, action_rng=None, shuffle_rng=None, restore_random=False):
    state = torch.load(path, map_location=policy.device, weights_only=False)
    if state["boundary"] != "complete_update" or state["architecture"] != policy.architecture:
        raise ValueError("Checkpoint architecture/boundary mismatch")
    if config is not None and state["config"] != config:
        raise ValueError("Resume configuration differs from checkpoint")
    if hashes is not None and state["hashes"] != hashes:
        raise ValueError("Resume data/source hashes differ from checkpoint")
    if restore_random and any(item is None for item in (task_rng, action_rng, shuffle_rng)):
        raise ValueError("Exact resume needs all RNG streams")
    policy.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_random:
        restore_rng(state["rng"])
        task_rng.bit_generator.state = state["task_rng"]
        action_rng.set_state(state["action_rng"].cpu())
        shuffle_rng.set_state(state["shuffle_rng"].cpu())
    return state
