"""Paper-described WeightNorm visual sizes, with explicit inherited GRU details.

WeightNorm uses w=g*v/||v|| independently per output unit. Embedding storage is
[action, feature], so its output-unit norm is over actions (columns normalized).
Initial g equals the initialized v norm; this is not data-dependent init.
"""
import torch
from torch import nn
from torch.nn import functional as F


class WeightNormWeight(nn.Module):
    def __init__(self, shape, output_axis=0):
        super().__init__()
        self.output_axis = output_axis
        self.norm_axes = tuple(i for i in range(len(shape)) if i != output_axis)
        self.v = nn.Parameter(torch.empty(*shape))
        gain_shape = [1] * len(shape)
        gain_shape[output_axis] = shape[output_axis]
        self.g = nn.Parameter(torch.empty(*gain_shape))
        nn.init.xavier_normal_(self.v)
        self.reset_gain()

    @torch.no_grad()
    def reset_gain(self):
        self.g.copy_(torch.linalg.vector_norm(self.v, dim=self.norm_axes, keepdim=True))

    def _apply(self, fn, recurse=True):
        # Cached tensors are deliberately not checkpoint buffers. Invalidate
        # them when moving/casting parameters, even if Parameter IDs survive.
        self._cache_key = None
        self._cached_weight = None
        return super()._apply(fn,recurse=recurse)

    def forward(self):
        cache_key = (id(self.v), self.v._version, id(self.g), self.g._version)
        cacheable = not self.training and not torch.is_grad_enabled()
        if cacheable and getattr(self, '_cache_key', None) == cache_key:
            return self._cached_weight
        norm = torch.linalg.vector_norm(self.v, dim=self.norm_axes, keepdim=True)
        weight = self.v * (self.g / norm.clamp_min(torch.finfo(self.v.dtype).tiny))
        if cacheable:
            self._cached_weight = weight.detach()
            self._cache_key = cache_key
        return weight


class WeightNormLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.normalized_weight = WeightNormWeight((output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

    @property
    def weight(self):
        return self.normalized_weight()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class WeightNormConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, stride=2):
        super().__init__()
        self.normalized_weight = WeightNormWeight((output_channels, input_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(output_channels))
        self.stride = stride

    @property
    def weight(self):
        return self.normalized_weight()

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride)


class WeightNormEmbedding(nn.Module):
    def __init__(self, actions=3, features=256):
        super().__init__()
        # F.embedding table is transposed relative to one-hot -> dense weights.
        self.normalized_weight = WeightNormWeight((actions, features), output_axis=1)

    @property
    def weight(self):
        return self.normalized_weight()

    def forward(self, actions):
        return F.embedding(actions, self.weight)


class WeightNormGRU(nn.Module):
    """ReLU reset-after GRU; the upstream gate constant is explicitly retained."""
    def __init__(self, input_dim=258, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._forget_bias = 1.0
        self._x2zr = WeightNormLinear(input_dim, 2 * hidden_dim)
        self._h2zr = WeightNormLinear(hidden_dim, 2 * hidden_dim)
        self._x2hhat = WeightNormLinear(input_dim, hidden_dim)
        self._h2hhat = WeightNormLinear(hidden_dim, hidden_dim)
        with torch.no_grad():
            for gate in self._h2zr.normalized_weight.v.chunk(2, 0):
                nn.init.orthogonal_(gate)
            nn.init.orthogonal_(self._h2hhat.normalized_weight.v)
            self._h2zr.normalized_weight.reset_gain()
            self._h2hhat.normalized_weight.reset_gain()

    def initial_state(self, batch_size):
        return self._h2hhat.normalized_weight.v.new_zeros(batch_size, self.hidden_dim)

    def forward(self, inputs, state=None, valid_mask=None, reset_mask=None):
        step = inputs.ndim == 2
        if step:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError('GRU expects nonempty [B,T,F] or [B,F] inputs')
        batch, timesteps = inputs.shape[:2]
        state = self.initial_state(batch) if state is None else state
        valid = torch.ones((batch, timesteps), dtype=torch.bool, device=inputs.device) if valid_mask is None else valid_mask.bool()
        reset = torch.zeros_like(valid) if reset_mask is None else reset_mask.bool()
        inputs = torch.where(valid[..., None], inputs, torch.zeros_like(inputs))
        gate_x = self._x2zr(inputs)
        candidate_x = self._x2hhat(inputs)
        # Normalize recurrent matrices once per sequence, not 500 times.
        recurrent_gates, recurrent_candidate = self._h2zr.weight, self._h2hhat.weight
        outputs = []
        for t in range(timesteps):
            state = torch.where((valid[:, t] & reset[:, t])[:, None], torch.zeros_like(state), state)
            update, reset_gate = (gate_x[:, t] + F.linear(state, recurrent_gates, self._h2zr.bias)).chunk(2, -1)
            update = torch.sigmoid(update - self._forget_bias)
            reset_gate = torch.sigmoid(reset_gate)
            candidate = F.relu(candidate_x[:, t] + reset_gate * F.linear(state, recurrent_candidate, self._h2hhat.bias))
            next_state = (1 - update) * state + update * candidate
            state = torch.where(valid[:, t, None], next_state, state)
            outputs.append(torch.where(valid[:, t, None], state, torch.zeros_like(state)))
        features = torch.stack(outputs, 1)
        return (features[:, 0] if step else features), state


class WeightNormVisualPolicy(nn.Module):
    def __init__(self, hidden_dim=256, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(WeightNormConv2d(3, 16), nn.ReLU(), WeightNormConv2d(16, 16), nn.ReLU())
        self.action_embedding = WeightNormEmbedding(3, 256)
        self.fusion = WeightNormLinear(816, 256)
        self.gru = WeightNormGRU(258, hidden_dim)
        self.policy_head = WeightNormLinear(hidden_dim, 3)
        self.value_head = WeightNormLinear(hidden_dim, 1)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def architecture(self):
        return {
            'protocol': 'rl2-task3-paper-weightnorm-v1', 'input': [3, 30, 40],
            'cnn': [{'in': 3, 'out': 16, 'kernel': 5, 'stride': 2, 'padding': 0},
                    {'in': 16, 'out': 16, 'kernel': 5, 'stride': 2, 'padding': 0}],
            'cnn_flatten': 560, 'embedding': [3, 256], 'fusion': [816, 256],
            'gru_input': 258, 'hidden_dim': self.hidden_dim, 'gru_candidate': 'relu',
            'gru_reset_after': True, 'forget_bias_constant': 1.0,
            'normalization': 'weight_norm_all_weight_matrices', 'layer_norm': False,
            'weight_norm': 'w=g*v/||v|| per output unit; embedding columns over action inputs',
            'initialization': 'Xavier normal feedforward/input; per-gate orthogonal recurrent; biases zero; g=initial_v_norm',
            'data_dependent_initialization': False, 'shared_actor_critic': True,
            'pretrained_weights': None, 'parameter_count': sum(p.numel() for p in self.parameters()),
            'inherited_details_not_confirmed_by_paper': ['reset-after GRU formula', 'forget-bias equation constant 1',
                                                       'zero padding in convolution', 'Xavier normal variant'],
        }

    def initial_state(self, batch_size):
        return self.gru.initial_state(batch_size)

    def _encode(self, images, prev_actions, prev_rewards, prev_done):
        images = images.to(self.device)
        images = images.float().div(127.5).sub(1) if images.dtype == torch.uint8 else images.float()
        if images.shape[-3:] != (3, 30, 40):
            raise ValueError('Expected RGB CHW 30x40 images')
        leading = images.shape[:-3]
        visual = self.cnn(images.reshape(-1, 3, 30, 40)).flatten(1).reshape(*leading, 560)
        actions = self.action_embedding(prev_actions.to(self.device, dtype=torch.long))
        fused = F.relu(self.fusion(torch.cat((visual, actions), -1)))
        return torch.cat((fused, prev_rewards.to(self.device).float()[..., None], prev_done.to(self.device).float()[..., None]), -1)

    def forward_step(self, image, prev_action, prev_reward, prev_done, h):
        features, state = self.gru(self._encode(image, prev_action, prev_reward, prev_done), h)
        return self.policy_head(features), self.value_head(features).squeeze(-1), state

    def static_forward_sequence(self, images, prev_actions, prev_rewards, prev_done, valid_mask, reset_mask):
        """Capture-safe forward; callers validate lengths/masks before capture."""
        valid = valid_mask.to(self.device).bool()
        images = torch.where(valid[..., None, None, None], images.to(self.device), torch.zeros_like(images, device=self.device))
        actions = torch.where(valid, prev_actions.to(self.device), torch.zeros_like(prev_actions, device=self.device))
        rewards = torch.where(valid, prev_rewards.to(self.device), torch.zeros_like(prev_rewards, device=self.device))
        dones = torch.where(valid, prev_done.to(self.device), torch.zeros_like(prev_done, device=self.device))
        features, state = self.gru(self._encode(images, actions, rewards, dones), self.initial_state(images.shape[0]),
                                   valid, reset_mask.to(self.device))
        logits, values = self.policy_head(features), self.value_head(features).squeeze(-1)
        return torch.where(valid[..., None], logits, torch.zeros_like(logits)), torch.where(valid, values, torch.zeros_like(values)), state

    def forward_sequence(self, images, prev_actions, prev_rewards, prev_done, lengths, reset_mask=None):
        batch, timesteps = images.shape[:2]
        lengths = lengths.to(self.device)
        if lengths.shape != (batch,) or torch.any(lengths <= 0) or torch.any(lengths > timesteps):
            raise ValueError('Every sequence must have a positive valid length')
        valid = torch.arange(timesteps, device=self.device)[None, :] < lengths[:, None]
        reset = torch.zeros_like(valid) if reset_mask is None else reset_mask.to(self.device)
        return self.static_forward_sequence(images, prev_actions, prev_rewards, prev_done, valid, reset)
