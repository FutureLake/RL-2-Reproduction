"""Optional fixed-shape CUDA graph for PPO forward and both backward passes.

Optimizer steps stay outside the graph. ``run`` owns stable gradient buffers:
callers must not zero gradients or call backward themselves. The graph performs
zeroing on every replay, so two PPO minibatches never accumulate gradients.
"""

import gc
import random

import numpy as np
import torch
from torch.distributions import Distribution


class CudaLossGraph:
    def __init__(self, loss_function, policy, value, sample, clip_param, ent_coef):
        # Unreachable eager autograd graphs can retain AccumulateGrad nodes tied
        # to an old stream. Callers must also release any live eager loss tensors
        # before capture (or construct this engine before the first backward).
        gc.collect()
        self.parameters = [p for net in (policy, value) for p in net.parameters() if p.requires_grad]
        if not self.parameters or not self.parameters[0].is_cuda:
            raise ValueError("CudaLossGraph requires CUDA policy/value parameters")
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in self.parameters):
            raise ValueError("All captured parameters must be on one CUDA device")
        if not sample or any(not isinstance(x, torch.Tensor) or x.device != self.device for x in sample.values()):
            raise ValueError("All sample fields must be CUDA tensors on the model device")
        self.static_data = {key: tensor.detach().clone() for key, tensor in sample.items()}
        self.gradients = [torch.zeros_like(p) for p in self.parameters]
        prior_gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters]
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        self.graph = torch.cuda.CUDAGraph()
        self._losses = None
        # Upstream stores its initial GRU state as an ordinary CPU tensor rather
        # than a registered buffer. Stage it only while capture records the
        # device recurrence, then restore the module attribute. Keep the staged
        # tensors alive because replay reads their fixed device addresses.
        original_initial_states = []
        self._initial_states = []
        for net in (policy, value):
            for module in net.modules():
                state = getattr(module, "_initial_state", None)
                if isinstance(state, torch.Tensor) and state.device != self.device:
                    converted = state.to(self.device)
                    original_initial_states.append((module, state))
                    self._initial_states.append(converted)
                    module._initial_state = converted
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        validate = Distribution._validate_args
        try:
            # Categorical validation otherwise materializes a device boolean on
            # the host, which CUDA stream capture explicitly forbids. Inputs and
            # finite losses/gradients are checked by the benchmark outside capture.
            Distribution.set_default_validate_args(False)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    for gradient in self.gradients:
                        gradient.zero_()
                    losses = loss_function(self.static_data, policy, value, clip_param, ent_coef)
                    losses["policy_loss"].backward()
                    losses["value_loss"].backward()
                del losses
            torch.cuda.current_stream(self.device).wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph, stream=warmup):
                for gradient in self.gradients:
                    gradient.zero_()
                losses = loss_function(self.static_data, policy, value, clip_param, ent_coef)
                losses["policy_loss"].backward()
                losses["value_loss"].backward()
            self._losses = {key: tensor.detach() for key, tensor in losses.items()}
            torch.cuda.current_stream(self.device).wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            # Construction does not perform an optimization step or leave warmup
            # gradients behind. Retain their addresses for all subsequent replay.
            for gradient, previous in zip(self.gradients, prior_gradients):
                if previous is None:
                    gradient.zero_()
                else:
                    gradient.copy_(previous)
        except Exception as exc:
            raise RuntimeError("CUDA loss graph capture failed; disable CUDA loss graphs to use eager PPO") from exc
        finally:
            for module, state in original_initial_states:
                module._initial_state = state
            Distribution.set_default_validate_args(validate)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

    def run(self, data):
        """Copy one matching minibatch, replay forward/backward, return detached losses."""
        if data.keys() != self.static_data.keys():
            raise ValueError("CUDA graph minibatch fields differ from captured fields")
        for key, tensor in data.items():
            target = self.static_data[key]
            if tensor.shape != target.shape or tensor.dtype != target.dtype or tensor.device != target.device:
                raise ValueError(f"CUDA graph field {key} differs in shape, dtype or device")
        if any(parameter.grad is not gradient for parameter, gradient in zip(self.parameters, self.gradients)):
            raise RuntimeError("Captured gradient buffers were replaced; do not call zero_grad(set_to_none=True)")
        for key, tensor in data.items():
            self.static_data[key].copy_(tensor)
        self.graph.replay()
        return self._losses
