"""Randomly initialized visual RL² actor-critic, with the reference GRU equations.

GRU equations adapted from lucaslingle/pytorch_rl2, commit
51afa4b6c6d05b469c7f74076b9490ae0809b289. That checkout has no LICENSE file;
no redistribution license is asserted here. The checkout is not modified.
Task 3 changes initialization, device handling and sequence masks.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceLayerNorm(nn.Module):
    def __init__(self, units):
        super().__init__()
        self._units = units
        self._g = nn.Parameter(torch.ones(units))
        self._b = nn.Parameter(torch.zeros(units))

    def forward(self, x):
        return F.layer_norm(x, (self._units,), self._g, self._b, 1e-8)


class Task3GRU(nn.Module):
    """Reference reset-after ReLU/LayerNorm GRU with stable sequence shapes."""

    def __init__(self, input_dim=258, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._forget_bias = 1.0  # Equation constant, not a trainable parameter bias.
        self._x2zr = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self._h2zr = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self._x2hhat = nn.Linear(input_dim, hidden_dim, bias=False)
        self._h2hhat = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._x2zr_ln = ReferenceLayerNorm(2 * hidden_dim)
        self._h2zr_ln = ReferenceLayerNorm(2 * hidden_dim)
        self._x2hhat_ln = ReferenceLayerNorm(hidden_dim)
        self._h2hhat_ln = ReferenceLayerNorm(hidden_dim)
        nn.init.xavier_normal_(self._x2zr.weight)
        nn.init.xavier_normal_(self._x2hhat.weight)
        for matrix in self._h2zr.weight.chunk(2, dim=0):
            nn.init.orthogonal_(matrix)
        nn.init.orthogonal_(self._h2hhat.weight)

    def initial_state(self, batch_size):
        return self._h2hhat.weight.new_zeros(batch_size, self.hidden_dim)

    def forward(self, inputs, state=None, valid_mask=None, reset_mask=None):
        is_step = inputs.ndim == 2
        if is_step:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError("GRU inputs must have nonempty [B,T,F] or [B,F] shape")
        batch, timesteps = inputs.shape[:2]
        state = self.initial_state(batch) if state is None else state
        valid = torch.ones((batch, timesteps), dtype=torch.bool, device=inputs.device) if valid_mask is None else valid_mask.bool()
        reset = torch.zeros_like(valid) if reset_mask is None else reset_mask.bool()
        inputs = torch.where(valid[..., None], inputs, torch.zeros_like(inputs))
        zr_x = self._x2zr_ln(self._x2zr(inputs))
        candidate_x = self._x2hhat_ln(self._x2hhat(inputs))
        outputs = []
        for t in range(timesteps):
            state = torch.where((reset[:, t] & valid[:, t])[:, None], torch.zeros_like(state), state)
            update, reset_gate = (zr_x[:, t] + self._h2zr_ln(self._h2zr(state))).chunk(2, dim=-1)
            update = torch.sigmoid(update - self._forget_bias)
            reset_gate = torch.sigmoid(reset_gate)
            candidate = F.relu(candidate_x[:, t] + reset_gate * self._h2hhat_ln(self._h2hhat(state)))
            next_state = (1 - update) * state + update * candidate
            state = torch.where(valid[:, t, None], next_state, state)
            outputs.append(torch.where(valid[:, t, None], state, torch.zeros_like(state)))
        features = torch.stack(outputs, dim=1)
        return (features[:, 0] if is_step else features), state


class VisualPolicy(nn.Module):
    def __init__(self, hidden_dim=256, device="cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, 5, 2), nn.ReLU(), nn.Conv2d(16, 16, 5, 2), nn.ReLU())
        self.action_embedding = nn.Embedding(3, 256)
        self.fusion = nn.Linear(560 + 256, 256)
        self.gru = Task3GRU(258, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        for module in (self.cnn[0], self.cnn[2], self.action_embedding, self.fusion, self.policy_head, self.value_head):
            nn.init.xavier_normal_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def architecture(self):
        return {"protocol": "rl2-task3-from-scratch-v2", "input": [3, 30, 40],
                "cnn": [{"in": 3, "out": 16, "kernel": 5, "stride": 2, "padding": 0},
                        {"in": 16, "out": 16, "kernel": 5, "stride": 2, "padding": 0}],
                "cnn_flatten": 560, "embedding": [3, 256], "fusion": [816, 256],
                "gru_input": 258, "hidden_dim": self.hidden_dim, "gru_candidate": "relu",
                "gru_reset_after": True, "layer_norm_epsilon": 1e-8,
                "layer_norm_gate_width": 2 * self.hidden_dim, "forget_bias_constant": 1.0,
                "initialization": "Xavier normal feedforward/input; per-gate orthogonal recurrent; trainable biases zero; LN gain one",
                "normalization": "LayerNorm, not paper WeightNorm", "shared_actor_critic": True,
                "pretrained_weights": None, "parameter_count": sum(p.numel() for p in self.parameters())}

    def initial_state(self, batch_size):
        return self.gru.initial_state(batch_size)

    def _encode(self, images, prev_actions, prev_rewards, prev_episode_done):
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float().div(127.5).sub(1)
        else:
            images = images.float()  # Floating inputs are already normalized; useful for gradient tests.
        if images.shape[-3:] != (3, 30, 40):
            raise ValueError("VisualPolicy expects RGB CHW 30x40 frames")
        leading = images.shape[:-3]
        features = self.cnn(images.reshape(-1, 3, 30, 40)).flatten(1).reshape(*leading, 560)
        actions = self.action_embedding(prev_actions.to(self.device, dtype=torch.long))
        fused = F.relu(self.fusion(torch.cat((features, actions), dim=-1)))
        return torch.cat((fused, prev_rewards.to(self.device).float().unsqueeze(-1),
                          prev_episode_done.to(self.device).float().unsqueeze(-1)), dim=-1)

    def forward_step(self, image, prev_action, prev_reward, prev_done, h):
        features, state = self.gru(self._encode(image, prev_action, prev_reward, prev_done), h)
        return self.policy_head(features), self.value_head(features).squeeze(-1), state

    def forward_sequence(self, images, prev_actions, prev_rewards, prev_episode_done, lengths, reset_mask=None):
        batch, timesteps = images.shape[:2]
        lengths = lengths.to(self.device)
        if lengths.shape != (batch,) or torch.any(lengths <= 0) or torch.any(lengths > timesteps):
            raise ValueError("Every sequence needs a valid positive length")
        valid = torch.arange(timesteps, device=self.device)[None, :] < lengths[:, None]
        images = torch.where(valid[..., None, None, None], images.to(self.device), torch.zeros_like(images, device=self.device))
        actions = torch.where(valid, prev_actions.to(self.device), torch.zeros_like(prev_actions, device=self.device))
        rewards = torch.where(valid, prev_rewards.to(self.device), torch.zeros_like(prev_rewards, device=self.device))
        dones = torch.where(valid, prev_episode_done.to(self.device), torch.zeros_like(prev_episode_done, device=self.device))
        features, state = self.gru(self._encode(images, actions, rewards, dones), self.initial_state(batch),
                                   valid, None if reset_mask is None else reset_mask.to(self.device))
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return torch.where(valid[..., None], logits, torch.zeros_like(logits)), torch.where(valid, values, torch.zeros_like(values)), state
