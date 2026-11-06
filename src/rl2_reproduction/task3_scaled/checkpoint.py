"""Atomic complete-update checkpoints for deterministic trial-index sampling."""
from pathlib import Path
import os
import torch

from rl2_reproduction.task3.utils import capture_rng, restore_rng
from .atomic_io import replace_with_retry


def save_checkpoint(path, policy, optimizer, config, hashes, *, update, training_steps,
                    next_trial_index, shuffle_rng, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config, "hashes": hashes, "architecture": policy.architecture,
             "update": int(update), "training_steps": int(training_steps),
             "next_trial_index": int(next_trial_index), "shuffle_rng": shuffle_rng.get_state(),
             "rng": capture_rng(), "boundary": "complete_update", "extra": extra or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)
    return state


def load_checkpoint(path, policy, optimizer=None, *, config=None, hashes=None,
                    shuffle_rng=None, restore_random=False):
    # Let load_state_dict move model/moments to parameter devices. AdamW's
    # non-capturable scalar step and RNG byte tensors must remain on CPU.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["boundary"] != "complete_update" or state["architecture"] != policy.architecture:
        raise ValueError("Checkpoint architecture or boundary differs")
    if config is not None and state["config"] != config:
        raise ValueError("Checkpoint configuration differs")
    if hashes is not None and state["hashes"] != hashes:
        raise ValueError("Checkpoint code/data provenance differs")
    if restore_random and shuffle_rng is None:
        raise ValueError("Exact resume requires shuffle RNG")
    policy.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_random:
        restore_rng(state["rng"])
        shuffle_rng.set_state(state["shuffle_rng"].cpu())
    return state
