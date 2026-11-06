"""Complete variable-length trials, padding masks and trial-boundary GAE."""

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class TrialRecord:
    arrays: dict
    task_id: str
    episode_summaries: list
    trace: list = field(default_factory=list)

    @property
    def length(self):
        return len(self.arrays["actions"])


def collate_trials(trials, device="cpu"):
    if not trials:
        raise ValueError("Cannot collate an empty trial batch")
    lengths = torch.tensor([t.length for t in trials], dtype=torch.long, device=device)
    maximum = int(lengths.max())
    batch = {"lengths": lengths, "mask": torch.arange(maximum, device=device)[None] < lengths[:, None]}
    fields = trials[0].arrays.keys()
    for key in fields:
        first = torch.as_tensor(trials[0].arrays[key])
        array = torch.zeros((len(trials), maximum, *first.shape[1:]), dtype=first.dtype, device=device)
        for i, trial in enumerate(trials):
            tensor = torch.as_tensor(trial.arrays[key], device=device)
            if tensor.shape[0] != trial.length:
                raise ValueError(f"Trial field {key} has wrong time length")
            array[i, :trial.length] = tensor
        batch[key] = array.detach()
    for i, trial in enumerate(trials):
        done = batch["trial_done"][i, :trial.length]
        if int(done.sum()) != 1 or not bool(done[-1]):
            raise ValueError("Batch must contain complete trials with only the final trial_done")
        if int(batch["episode_done"][i, :trial.length].sum()) != 2:
            raise ValueError("Task 3 requires exactly two complete episodes per trial")
    return batch


@torch.no_grad()
def compute_gae(batch, gamma=0.99, gae_lambda=0.99):
    """Episode resets do not terminate credit; only the true trial end does."""
    rewards, values, mask = batch["rewards"], batch["old_values"], batch["mask"]
    advantages = torch.zeros_like(rewards)
    future_advantage = torch.zeros(rewards.shape[0], device=rewards.device)
    future_value = torch.zeros_like(future_advantage)
    for t in reversed(range(rewards.shape[1])):
        live = mask[:, t]
        continuation = (~batch["trial_done"][:, t].bool()).float()
        delta = rewards[:, t] + gamma * continuation * future_value - values[:, t]
        current = delta + gamma * gae_lambda * continuation * future_advantage
        advantages[:, t] = torch.where(live, current, torch.zeros_like(current))
        future_advantage = torch.where(live, current, torch.zeros_like(current))
        future_value = torch.where(live, values[:, t], torch.zeros_like(current))
    batch["advantages"] = advantages.detach()
    batch["return_targets"] = torch.where(mask, advantages + values, torch.zeros_like(values)).detach()
    return batch


def select_trials(batch, indices):
    """Minibatches preserve each complete trial's temporal order."""
    return {key: value[indices] for key, value in batch.items()}
