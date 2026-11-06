"""Deterministic full-Trial CPU actors, with bounded on-policy collection.

Only the execution schedule is parallel. Trial IDs determine task and action
streams, independently of worker assignment or completion time. Old Task 3
environment/collector code is reused without modification.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback

import numpy as np
import torch

from ..task3.trial_collector import TrialCollector


MAX_TRIAL_ACTIONS = 500
_POLICY = None
_ENV = None
_WEIGHT_DIGEST = None


def state_dict_digest(state_dict):
    """Hash parameter names, dtype, shape and exact values on the host."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trial_spec(seed, trial_index, task_count):
    """Stable independent task/action streams; restarting needs only next ID."""
    if type(trial_index) is not int or trial_index < 0 or task_count <= 0:
        raise ValueError("Trial index and task count must be nonnegative/positive")
    task_rng = np.random.default_rng(np.random.SeedSequence([seed, 501, trial_index]))
    action_seed = int(np.random.SeedSequence([seed, 502, trial_index]).generate_state(1, dtype=np.uint64)[0])
    return int(task_rng.integers(task_count)), action_seed


def wave_size(remaining, workers):
    """No wave prefix can reach target before the final dispatched trial.

    Every Trial has at most 500 actions. Near the target, dispatch one Trial.
    Therefore all sampled actions are committed and overshoot is at most 499.
    """
    if type(remaining) is not int or remaining <= 0:
        raise ValueError("remaining must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    return min(workers, max(1, remaining // MAX_TRIAL_ACTIONS))


class _CountedEnvironment:
    def __init__(self, environment, counter):
        self._environment = environment
        self._counter = counter

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def step(self, action):
        result = self._environment.step(action)
        with self._counter.get_lock():
            self._counter.value += 1
        return result


def _initialize_worker(calibration, ready, counter):
    global _POLICY, _ENV, _WEIGHT_DIGEST
    # This spawn child is strictly a CPU actor. Set before any CUDA query or
    # seed setup, so evaluation helpers can never create a GPU context here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("A CPU actor entered initialization with an existing CUDA context")
        from multiprocessing.util import Finalize
        from ..task3.env import MazeEpisodeEnv
        from ..task3.utils import configure
        from .model import WeightNormVisualPolicy
        configure(0, threads=1)
        torch.set_num_interop_threads(1)
        _POLICY = WeightNormVisualPolicy(hidden_dim=256, device="cpu").eval()
        _ENV = _CountedEnvironment(MazeEpisodeEnv(calibration=calibration), counter)
        _WEIGHT_DIGEST = None
        Finalize(_ENV, _ENV.close, exitpriority=10)
        ready.put({"pid": os.getpid(), "ready": True,
                   "initialization_seconds": time.perf_counter() - started,
                   "torch_threads": torch.get_num_threads(),
                   "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "cuda_initialized": torch.cuda.is_initialized()})
    except BaseException:
        ready.put({"pid": os.getpid(), "ready": False, "traceback": traceback.format_exc()})
        raise


def _ensure_weights(state_dict, digest):
    global _WEIGHT_DIGEST
    if _POLICY is None or _ENV is None:
        raise RuntimeError("CPU actor was not initialized")
    if digest != _WEIGHT_DIGEST:
        actual = state_dict_digest(state_dict)
        if actual != digest:
            raise RuntimeError(f"Actor weight transport hash mismatch: {actual} != {digest}")
        _POLICY.load_state_dict(state_dict, strict=True)
        _POLICY.eval()
        loaded = state_dict_digest(_POLICY.state_dict())
        if loaded != digest:
            raise RuntimeError("Actor loaded weights do not match the frozen policy snapshot")
        _WEIGHT_DIGEST = digest


def _collect_one(job):
    started = time.perf_counter()
    _ensure_weights(job["state_dict"], job["weight_digest"])
    generator = torch.Generator(device="cpu").manual_seed(job["action_seed"])
    trial = TrialCollector(_ENV, _POLICY, generator).collect(job["task"], mode="rl2")
    if not 2 <= trial.length <= MAX_TRIAL_ACTIONS:
        raise RuntimeError("Actor violated complete two-Episode length bounds")
    if int(trial.arrays["episode_done"].sum()) != 2 or int(trial.arrays["trial_done"].sum()) != 1:
        raise RuntimeError("Actor did not return one complete two-Episode Trial")
    return {"trial": trial, "trial_index": job["trial_index"], "pid": os.getpid(),
            "weight_digest": _WEIGHT_DIGEST, "action_seed": job["action_seed"],
            "seconds": time.perf_counter() - started}


class ParallelCollector:
    def __init__(self, calibration, workers=4, seed=0, worker_timeout=180.0):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("Expected 1–8 CPU workers")
        self.calibration = str(Path(calibration).resolve()) if isinstance(calibration, (str, Path)) else calibration
        self.workers, self.seed = workers, int(seed)
        self.worker_timeout = float(worker_timeout)
        self._context = mp.get_context("spawn")
        self._counter = self._context.Value("q", 0)
        self._pool = None
        self._ready = None
        self._state_dict = None
        self.weight_digest = None
        self.initialization = []
        self._worker_pids = set()
        self.startup_seconds = 0.0

    @property
    def physical_actions(self):
        with self._counter.get_lock():
            return int(self._counter.value)

    def _start(self):
        if self._pool is not None:
            return
        started = time.perf_counter()
        self._ready = self._context.Queue()
        self._pool = self._context.Pool(self.workers, initializer=_initialize_worker,
                                        initargs=(self.calibration, self._ready, self._counter))
        try:
            self.initialization = []
            for _ in range(self.workers):
                remaining = self.worker_timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("CPU actor initialization exceeded timeout")
                try:
                    row = self._ready.get(timeout=remaining)
                except queue.Empty as error:
                    raise TimeoutError("CPU actor initialization did not complete") from error
                self.initialization.append(row)
                if not row["ready"]:
                    raise RuntimeError(f"CPU actor initialization failed: {row}")
            if len({row["pid"] for row in self.initialization}) != self.workers:
                raise RuntimeError("CPU actor initialization did not yield unique workers")
            self._worker_pids = {row["pid"] for row in self.initialization}
        except BaseException:
            self.close(force=True)
            raise
        self.startup_seconds = time.perf_counter() - started

    def set_weights(self, state_dict_cpu):
        if not state_dict_cpu:
            raise ValueError("A complete policy state_dict is required")
        # Own immutable shared buffers. Caller mutation cannot alter a live batch.
        snapshot = {key: value.detach().cpu().contiguous().clone().share_memory_()
                    for key, value in state_dict_cpu.items()}
        if any(not torch.isfinite(value).all() for value in snapshot.values()):
            raise FloatingPointError("Cannot dispatch nonfinite actor weights")
        self._state_dict = snapshot
        self.weight_digest = state_dict_digest(snapshot)
        return self.weight_digest

    def _dispatch(self, function, jobs, progress=None):
        self._start()
        started = time.perf_counter()
        physical_before = self.physical_actions
        results = []
        last_progress = started
        try:
            if {worker.pid for worker in self._pool._pool} != self._worker_pids:
                raise RuntimeError("A CPU actor exited or was replaced before dispatch")
            futures = [self._pool.apply_async(function, (job,)) for job in jobs]
            for future in futures:
                while True:
                    try:
                        results.append(future.get(timeout=1.0))
                        break
                    except mp.TimeoutError:
                        now = time.perf_counter()
                        if {worker.pid for worker in self._pool._pool} != self._worker_pids:
                            raise RuntimeError("A CPU actor exited or was replaced during a Trial")
                        if now - started > self.worker_timeout:
                            raise TimeoutError(f"CPU actor wave exceeded {self.worker_timeout}s")
                        if progress is not None and now - last_progress >= 5.0:
                            progress({"actor_phase": "actor_wave", "completed_in_wave": len(results),
                                      "wave_trials": len(jobs),
                                      "physical_actions_in_wave": self.physical_actions - physical_before,
                                      "seconds": now - started})
                            last_progress = now
            if {worker.pid for worker in self._pool._pool} != self._worker_pids:
                raise RuntimeError("A CPU actor exited or was replaced during a Trial")
        except BaseException as error:
            sampled = self.physical_actions - physical_before
            # Kill only owned actors and their native-engine children on hard failure.
            self.close(force=True)
            raise RuntimeError(f"Actor wave failed after {sampled} completed physical actions; "
                               f"no partial Trial is committed. Cause: {error}") from error
        return results

    def collect(self, tasks, target_steps, start_trial_index=0, progress=None):
        if type(target_steps) is not int or target_steps < 0:
            raise ValueError("target_steps must be a nonnegative integer")
        if type(start_trial_index) is not int or start_trial_index < 0:
            raise ValueError("start_trial_index must be a nonnegative integer")
        if not tasks or any(task.get("split") != "train" for task in tasks):
            raise ValueError("Training collection accepts only nonempty training tasks")
        if self._state_dict is None:
            raise RuntimeError("set_weights must precede collection")
        started = time.perf_counter()
        before = self.physical_actions
        trials, worker_rows = [], []
        total, next_index = 0, start_trial_index
        while total < target_steps:
            count = wave_size(target_steps - total, self.workers)
            jobs = []
            for index in range(next_index, next_index + count):
                task_index, action_seed = trial_spec(self.seed, index, len(tasks))
                jobs.append({"task": tasks[task_index], "trial_index": index,
                             "action_seed": action_seed, "state_dict": self._state_dict,
                             "weight_digest": self.weight_digest})
            rows = self._dispatch(_collect_one, jobs, progress)
            for expected, row in zip(jobs, rows):
                if row["trial_index"] != expected["trial_index"] or row["weight_digest"] != self.weight_digest:
                    raise RuntimeError("Actor returned a different Trial ID or policy version")
                trials.append(row["trial"])
                total += row["trial"].length
                worker_rows.append({key: value for key, value in row.items() if key != "trial"})
            next_index += count
            if progress is not None:
                progress({"actor_phase": "collection", "completed_trials": len(trials), "target_steps": target_steps,
                          "env_actions": total, "physical_actions": self.physical_actions - before,
                          "next_trial_index": next_index, "seconds": time.perf_counter() - started})
        physical = self.physical_actions - before
        if physical != total or (target_steps and not target_steps <= total <= target_steps + 499):
            raise RuntimeError(f"Collection accounting mismatch: physical={physical}, committed={total}, target={target_steps}")
        return trials, {"next_trial_index": next_index, "first_trial_index": start_trial_index,
                        "env_actions": total, "physical_actions": physical, "discarded_actions": 0,
                        "trial_count": len(trials), "target_steps": target_steps,
                        "overshoot": total - target_steps, "workers": self.workers,
                        "seconds": time.perf_counter() - started, "weight_digest": self.weight_digest,
                        "task_ids": [trial.task_id for trial in trials], "worker_trials": worker_rows,
                        "worker_initialization": self.initialization,
                        "rng_scheme": "SeedSequence(seed,501,trial_index) task; SeedSequence(seed,502,trial_index) actions"}

    def close(self, force=False):
        pool, self._pool = self._pool, None
        if pool is not None:
            if force:
                if os.name == "nt":
                    # Prevent Pool's handler from replacing actors while their
                    # process trees are being stopped. terminate() subsequently
                    # handles queues and every remaining owned process.
                    pool._worker_handler._state = mp.pool.TERMINATE
                    pool._change_notifier.put(None)
                    pool._worker_handler.join(timeout=5.0)
                    for worker in pool._pool:
                        if worker.pid and worker.is_alive():
                            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                pool.terminate()
            else:
                pool.close()
            pool.join()
        if self._ready is not None:
            self._ready.close()
            self._ready.join_thread()
            self._ready = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close(force=exc_type is not None)
