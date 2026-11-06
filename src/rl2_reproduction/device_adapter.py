"""Scoped device/throughput adaptations for the unchanged upstream network.

The GRU equations, ReLU candidate, LayerNorm epsilon, forget bias, parameter
names and state dictionaries are preserved. The context changes callable
implementations only and restores them even if training raises an exception.
It is intended for one controlled benchmark in a process, not concurrent threads.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F


def _device_one_hot(ys, depth):
    return F.one_hot(ys.long(), num_classes=depth).float()


def _device_initial_state(self, batch_size):
    parameter = self._h2hhat.weight
    return self._initial_state.to(device=parameter.device, dtype=parameter.dtype).unsqueeze(0).repeat(batch_size, 1)


def _fused_layer_norm(self, inputs, eps=1e-8):
    return F.layer_norm(inputs, (self._units,), self._g, self._b, eps)


def _precomputed_gru_forward(self, inputs, prev_state):
    """Same reference recurrence, with time-independent input work batched."""
    assert inputs.ndim in (2, 3)
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(1)
    timesteps = inputs.shape[1]
    zr_input = self._x2zr(inputs)
    candidate_input = self._x2hhat(inputs)
    if self._use_ln:
        zr_input = self._x2zr_ln(zr_input)
        candidate_input = self._x2hhat_ln(candidate_input)
    features = []
    state = prev_state
    for t in range(timesteps):
        zr_hidden = self._h2zr(state)
        if self._use_ln:
            zr_hidden = self._h2zr_ln(zr_hidden)
        update, reset = torch.chunk(zr_input[:, t] + zr_hidden, 2, dim=-1)
        update = torch.sigmoid(update - self._forget_bias)
        reset = torch.sigmoid(reset)
        if self._reset_after:
            candidate_hidden = self._h2hhat(state)
            if self._use_ln:
                candidate_hidden = self._h2hhat_ln(candidate_hidden)
            candidate = candidate_input[:, t] + reset * candidate_hidden
        else:
            candidate_hidden = self._h2hhat(reset * state)
            if self._use_ln:
                candidate_hidden = self._h2hhat_ln(candidate_hidden)
            candidate = candidate_input[:, t] + candidate_hidden
        state = (1 - update) * state + update * F.relu(candidate)
        features.append(state)
    features = torch.stack(features, dim=1)
    if timesteps == 1:
        features = features.squeeze(1)
    return features, state


@contextmanager
def device_adapters(fused_layer_norm=True, precompute_inputs=True):
    """Adapt tabular preprocessing and the reference GRU for CPU/CUDA use.

    Call after making the upstream ``rl2`` package importable. This does not
    change the CPU/NumPy PPO/MPI functions; the benchmark owns device-aware
    rollout and loss tensors. Setting either optimization flag to False retains
    its original implementation for parity checks and diagnostics.
    """
    from rl2.agents.preprocessing import tabular
    from rl2.agents.architectures.gru import GRU
    from rl2.agents.architectures.common.normalization import LayerNorm

    original_one_hot = tabular.one_hot
    original_initial_state = GRU.initial_state
    original_layer_norm = LayerNorm.forward
    original_gru_forward = GRU.forward
    try:
        tabular.one_hot = _device_one_hot
        GRU.initial_state = _device_initial_state
        if fused_layer_norm:
            LayerNorm.forward = _fused_layer_norm
        if precompute_inputs:
            GRU.forward = _precomputed_gru_forward
        yield
    finally:
        tabular.one_hot = original_one_hot
        GRU.initial_state = original_initial_state
        LayerNorm.forward = original_layer_norm
        GRU.forward = original_gru_forward
