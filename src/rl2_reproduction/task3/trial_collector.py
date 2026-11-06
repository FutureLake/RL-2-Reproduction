"""Single-environment collector of exactly two complete episodes per trial."""

from dataclasses import dataclass
import numpy as np
import torch
from .sequence_batch import TrialRecord


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass
class EpisodeRecord:
    arrays: dict
    summary: dict
    trace: list
    hidden: torch.Tensor
    feedback: tuple


def join_episodes(episodes, task_id):
    if len(episodes) != 2:
        raise ValueError("Task 3 trials require exactly two episodes")
    arrays = {key: np.concatenate([episode.arrays[key] for episode in episodes], axis=0)
              for key in episodes[0].arrays}
    arrays["trial_done"][:] = False
    arrays["trial_done"][-1] = True
    return TrialRecord(arrays, str(task_id), [episode.summary for episode in episodes],
                       [row for episode in episodes for row in episode.trace])


class TrialCollector:
    def __init__(self, env, policy, action_generator=None):
        self.env, self.policy = env, policy
        self.action_generator = action_generator or torch.Generator(device="cpu").manual_seed(0)

    @torch.no_grad()
    def collect_episode(self, task_spec, episode_index, h=None, feedback=(0, 0.0, 0.0), reset_hidden=False, generator=None):
        """Reset once, then stop at one real episode boundary; never auto-reset."""
        generator = generator or self.action_generator
        image, start_info = self.env.reset_episode(task_spec)
        if image is None:
            raise RuntimeError("reset_episode returned no image")
        h = self.policy.initial_state(1) if h is None else h.detach().clone()
        if reset_hidden:
            h = torch.zeros_like(h)
        previous_action, previous_reward, previous_done = feedback
        fields = {name: [] for name in ("images_uint8", "prev_actions", "prev_rewards", "prev_episode_done", "actions", "rewards",
                  "episode_done", "trial_done", "old_log_probs", "old_values", "old_logits", "episode_index", "reset_mask")}
        trace, total_reward, collisions, distance, turns = [], 0.0, 0, 0.0, 0
        max_steps = getattr(self.env, "max_steps_per_episode", 250)
        for t in range(max_steps + 1):
            if t == max_steps:
                raise RuntimeError("Environment failed to terminate at its declared episode limit")
            frame = np.asarray(image)
            if frame.dtype != np.uint8 or frame.shape != (3, 30, 40):
                raise ValueError("Environment must return uint8 RGB CHW [3,30,40]")
            inputs = (torch.as_tensor(frame.copy(), device=self.policy.device).unsqueeze(0),
                      torch.tensor([previous_action], dtype=torch.long, device=self.policy.device),
                      torch.tensor([previous_reward], dtype=torch.float32, device=self.policy.device),
                      torch.tensor([previous_done], dtype=torch.float32, device=self.policy.device))
            logits, value, next_h = self.policy.forward_step(*inputs, h)
            if not torch.isfinite(logits).all() or not torch.isfinite(value).all() or not torch.isfinite(next_h).all():
                raise FloatingPointError("Non-finite visual rollout")
            distribution = torch.distributions.Categorical(logits=logits)
            # One uniform draw per step permits paired evaluation despite different actions.
            uniform = torch.rand((), generator=generator, device=generator.device).cpu()
            action = int(torch.searchsorted(distribution.probs[0].cpu().cumsum(-1), uniform).clamp(max=2))
            new_image, reward, done, info = self.env.step(action)
            reward = float(reward)
            if not np.isfinite(reward):
                raise FloatingPointError("Non-finite environment reward")
            values = {"images_uint8": frame.copy(), "prev_actions": previous_action, "prev_rewards": previous_reward,
                      "prev_episode_done": previous_done, "actions": action, "rewards": reward,
                      "episode_done": bool(done), "trial_done": False,
                      "old_log_probs": float(distribution.log_prob(torch.tensor([action], device=self.policy.device))[0]),
                      "old_values": float(value[0]), "old_logits": logits[0].cpu().numpy().copy(),
                      "episode_index": episode_index, "reset_mask": bool(t == 0 and (episode_index == 0 or reset_hidden))}
            for name, item in values.items():
                fields[name].append(item)
            collided = bool(info.get("collided", False))
            step_distance = float(info.get("distance", 0.0))
            collisions += int(collided)
            distance += step_distance
            turns += int(action != 2)
            total_reward += reward
            trace.append({"episode_index": episode_index, "step": t + 1, "action": action, "reward": reward,
                          "done": bool(done), "pose": _plain(info.get("pose")), "success": bool(info.get("success", False)),
                          "collided": collided, "distance": step_distance, "prev_action": previous_action,
                          "prev_reward": float(previous_reward), "prev_done": float(previous_done),
                          "hidden_norm_before": float(h.norm()), "hidden_norm_after": float(next_h.norm())})
            previous_action, previous_reward, previous_done = action, reward, float(done)
            h = next_h
            if done:
                break
            if new_image is None:
                raise RuntimeError("Nonterminal environment step returned no image")
            image = new_image
        arrays = {}
        for name, items in fields.items():
            dtype = np.uint8 if name == "images_uint8" else np.int64 if name in ("actions", "prev_actions", "episode_index") else np.bool_ if name in ("episode_done", "trial_done", "reset_mask") else np.float32
            arrays[name] = np.asarray(items, dtype=dtype)
        summary = {"episode_index": episode_index, "return": total_reward, "length": len(trace),
                   "success": bool(info.get("success", False)), "collision_count": collisions, "physical_distance": distance,
                   "turn_count": turns, "turn_fraction": turns / len(trace), "goal_hits": int(bool(info.get("success", False))),
                   "start_pose": _plain(start_info.get("pose")), "end_pose": _plain(info.get("pose"))}
        return EpisodeRecord(arrays, summary, trace, h.detach(), (previous_action, previous_reward, previous_done))

    def collect(self, task_spec, mode="rl2", episodes=2):
        if episodes != 2 or mode not in ("rl2", "b1"):
            raise ValueError("Only two-episode rl2/b1 trials are supported in v2")
        first = self.collect_episode(task_spec, 0)
        second = self.collect_episode(task_spec, 1, first.hidden, first.feedback, reset_hidden=(mode == "b1"))
        task_id = task_spec.get("task_id", "unknown") if isinstance(task_spec, dict) else task_spec.task_id
        return join_episodes([first, second], task_id)
