"""Batched paper-distribution tasks and paired held-out policy evaluation.

The sampler has the same distributions and episode semantics as the reference
environments. NumPy's modern generator changes seeded realizations, not the task
distribution. Task generation, environment noise and action sampling are separate.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


class VectorTasks:
    """A synchronized batch of independent tasks, fixed for one RL2 trial.

    ``step`` auto-resets episode states but never resamples task parameters.
    Construct a new instance for a new batch of trials. No task parameters are
    included in observations; the policy only sees the ordinary state index.
    """

    def __init__(self, environment: str, num_envs: int, num_states: int,
                 num_actions: int, episode_length: int, task_seed: int):
        if environment not in ("bandit", "tabular_mdp"):
            raise ValueError(f"Unsupported environment: {environment}")
        if min(num_envs, num_states, num_actions, episode_length) <= 0:
            raise ValueError("Task dimensions and episode_length must be positive")
        self.environment = environment
        self.num_envs = int(num_envs)
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.episode_length = 1 if environment == "bandit" else int(episode_length)
        self.task_seed = int(task_seed)
        task_rng = np.random.default_rng(self.task_seed)
        if environment == "bandit":
            self.payout_probabilities = task_rng.uniform(0, 1, (num_envs, num_actions))
        else:
            self.reward_means = task_rng.normal(1, 1, (num_envs, num_states, num_actions))
            self.transition_probabilities = task_rng.dirichlet(
                np.ones(num_states), size=(num_envs, num_actions, num_states))
        self._initialize_noise(self.task_seed)
        self.reset()

    def _initialize_noise(self, seed: int):
        self._noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4E4F4953]))
        self._uniform_schedule = None
        self._normal_schedule = None
        self._noise_timestep = 0

    def reset(self) -> np.ndarray:
        self.states = np.zeros(self.num_envs, dtype=np.int64)
        self.elapsed = 0
        return self.states.copy()

    def _parameter_arrays(self):
        if self.environment == "bandit":
            return (self.payout_probabilities,)
        return self.reward_means, self.transition_probabilities

    @property
    def task_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([self.environment, self.num_envs, self.num_states,
                                  self.num_actions, self.episode_length]).encode())
        for array in self._parameter_arrays():
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    @property
    def task_fingerprints(self) -> list[str]:
        result = []
        for i in range(self.num_envs):
            digest = hashlib.sha256()
            digest.update(json.dumps([self.environment, self.num_states,
                                      self.num_actions, self.episode_length]).encode())
            for array in self._parameter_arrays():
                digest.update(np.ascontiguousarray(array[i]).tobytes())
            result.append(digest.hexdigest())
        return result

    def clone_slice(self, start: int, stop: int, noise_seed: int | None = None):
        """Copy a task subset, with reset episodes and independently reset noise."""
        if not (0 <= start < stop <= self.num_envs):
            raise ValueError("Invalid task subset")
        child = object.__new__(VectorTasks)
        for name in ("environment", "num_states", "num_actions", "episode_length", "task_seed"):
            setattr(child, name, getattr(self, name))
        child.num_envs = stop - start
        for name in (("payout_probabilities",) if self.environment == "bandit" else
                     ("reward_means", "transition_probabilities")):
            setattr(child, name, getattr(self, name)[start:stop].copy())
        child._initialize_noise(self.task_seed if noise_seed is None else noise_seed)
        child.reset()
        return child

    def set_noise_schedule(self, uniforms: np.ndarray, normals: np.ndarray | None = None):
        """Supply common per-step noise for paired evaluation (not observations)."""
        if uniforms.ndim != 2 or uniforms.shape[1] != self.num_envs:
            raise ValueError("uniforms must have shape [steps, num_envs]")
        if self.environment == "tabular_mdp" and (normals is None or normals.shape != uniforms.shape):
            raise ValueError("MDP normal noise must match the uniform schedule")
        self._uniform_schedule = uniforms
        self._normal_schedule = normals
        self._noise_timestep = 0

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be an integer array of shape [num_envs]")
        if np.any(actions < 0) or np.any(actions >= self.num_actions):
            raise ValueError("Action index outside action space")
        rows = np.arange(self.num_envs)
        if self._uniform_schedule is None:
            uniforms = self._noise_rng.random(self.num_envs)
            normals = self._noise_rng.standard_normal(self.num_envs) if self.environment == "tabular_mdp" else None
        else:
            if self._noise_timestep >= len(self._uniform_schedule):
                raise ValueError("Environment noise schedule exhausted")
            uniforms = self._uniform_schedule[self._noise_timestep]
            normals = None if self._normal_schedule is None else self._normal_schedule[self._noise_timestep]
        self._noise_timestep += 1
        if self.environment == "bandit":
            rewards = (uniforms < self.payout_probabilities[rows, actions]).astype(np.float32)
        else:
            # Reward is generated from the pre-transition state, as in upstream.
            rewards = (self.reward_means[rows, self.states, actions] + normals).astype(np.float32)
            probabilities = self.transition_probabilities[rows, actions, self.states]
            next_states = (uniforms[:, None] > probabilities.cumsum(axis=-1)).sum(axis=-1)
            self.states = np.minimum(next_states, self.num_states - 1).astype(np.int64)
        self.elapsed += 1
        done = self.elapsed >= self.episode_length
        if done:
            self.states.fill(0)
            self.elapsed = 0
        return self.states.copy(), rewards, np.full(self.num_envs, float(done), dtype=np.float32)


def _statistics(returns: np.ndarray) -> dict[str, Any]:
    returns = np.asarray(returns, dtype=np.float64)
    std = float(returns.std(ddof=1))
    sem = std / np.sqrt(len(returns))
    mean = float(returns.mean())
    return {"mean_return": mean, "std_return": std, "sem_return": float(sem),
            "ci95_low": float(mean - 1.96 * sem), "ci95_high": float(mean + 1.96 * sem),
            "returns": returns.tolist()}


def _to_device(state, device):
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, tuple):
        return tuple(_to_device(x, device) for x in state)
    if isinstance(state, list):
        return [_to_device(x, device) for x in state]
    if isinstance(state, dict):
        return {key: _to_device(value, device) for key, value in state.items()}
    return state


@torch.no_grad()
def evaluate(policy, config: dict, task_seed: int, action_seed: int,
             num_trials: int, device="cpu", batch_size: int = 100) -> dict[str, Any]:
    """Paired stochastic policy/uniform-random evaluation on identical fresh tasks.

    CIs describe variability across held-out tasks, conditional on one trained
    model. They do not estimate variability across training seeds. Common noise
    is applied to policy/random runs to reduce variance without revealing any
    environment parameters to either policy.
    """
    if num_trials < 2 or batch_size < 1:
        raise ValueError("num_trials >= 2 and batch_size >= 1 are required")
    device = torch.device(device)
    horizon = int(config["meta_episode_len"])
    episode_length = 1 if config["environment"] == "bandit" else int(config["max_episode_len"])
    if horizon < 1 or horizon % episode_length:
        raise ValueError("Evaluation requires a positive whole number of episodes")
    tasks = VectorTasks(config["environment"], num_trials, config["num_states"],
                        config["num_actions"], episode_length, task_seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([int(task_seed), 0x4556414C]))
    uniforms = noise_rng.random((horizon, num_trials))
    normals = noise_rng.standard_normal((horizon, num_trials)) if tasks.environment == "tabular_mdp" else None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(action_seed))
    random_actions = np.random.default_rng(np.random.SeedSequence([int(action_seed), 0x52414E44]))
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    training = policy.training
    parameters_before = [p.detach().clone() for p in policy.parameters()]
    episode_returns = {kind: np.zeros((num_trials, horizon // episode_length), dtype=np.float64)
                       for kind in ("policy", "random")}
    try:
        policy.eval()
        for start in range(0, num_trials, batch_size):
            stop = min(start + batch_size, num_trials)
            size = stop - start
            for kind in ("policy", "random"):
                env = tasks.clone_slice(start, stop)
                env.set_noise_schedule(uniforms[:, start:stop], None if normals is None else normals[:, start:stop])
                observations = env.reset()
                state = _to_device(policy.initial_state(size), device) if kind == "policy" else None
                previous_actions = np.zeros(size, dtype=np.int64)
                previous_rewards = np.zeros(size, dtype=np.float32)
                previous_dones = np.ones(size, dtype=np.float32)
                for t in range(horizon):
                    if kind == "policy":
                        distribution, state = policy(
                            curr_obs=torch.as_tensor(observations, dtype=torch.long, device=device),
                            prev_action=torch.as_tensor(previous_actions, dtype=torch.long, device=device),
                            prev_reward=torch.as_tensor(previous_rewards, dtype=torch.float32, device=device),
                            prev_done=torch.as_tensor(previous_dones, dtype=torch.float32, device=device),
                            prev_state=state)
                        actions = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1).cpu().numpy()
                    else:
                        actions = random_actions.integers(0, config["num_actions"], size=size, dtype=np.int64)
                    observations, rewards, dones = env.step(actions)
                    episode_returns[kind][start:stop, t // episode_length] += rewards
                    previous_actions, previous_rewards, previous_dones = actions, rewards, dones
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        policy.train(training)
    if not all(torch.equal(before, after) for before, after in zip(parameters_before, policy.parameters())):
        raise RuntimeError("Evaluation changed policy parameters")
    policy_returns = episode_returns["policy"].sum(axis=1)
    random_returns = episode_returns["random"].sum(axis=1)
    return {
        "task_seed": int(task_seed), "action_seed": int(action_seed), "num_trials": int(num_trials),
        "task_fingerprint": tasks.task_fingerprint, "task_fingerprints": tasks.task_fingerprints,
        "paired_tasks_identical": True, "weights_frozen": True,
        "confidence_interval_method": "normal approximation across held-out tasks, conditional on one trained model",
        "policy": _statistics(policy_returns), "random": _statistics(random_returns),
        "paired_difference": _statistics(policy_returns - random_returns),
        "policy_episode_returns": episode_returns["policy"].tolist(),
        "random_episode_returns": episode_returns["random"].tolist(),
        "policy_mean_episode_returns": episode_returns["policy"].mean(axis=0).tolist(),
        "random_mean_episode_returns": episode_returns["random"].mean(axis=0).tolist(),
    }
