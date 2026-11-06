"""Device adaptation must preserve original forward and gradient semantics."""

from copy import deepcopy

import pytest
import torch

from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.common.normalization import LayerNorm
from rl2.agents.preprocessing import tabular
from rl2_reproduction.device_adapter import device_adapters


def _inputs(device="cpu", sequence=True):
    generator = torch.Generator().manual_seed(801)
    shape = (3, 6) if sequence else (3,)
    return {"curr_obs": torch.randint(0, 4, shape, generator=generator).to(device),
            "prev_action": torch.randint(0, 3, shape, generator=generator).to(device),
            "prev_reward": torch.randn(shape, generator=generator).to(device),
            "prev_done": torch.randint(0, 2, shape, generator=generator).float().to(device)}


def _forward_backward(net, inputs, kind):
    output, state = net(**inputs, prev_state=net.initial_state(3))
    features = output.probs if kind == "policy" else output
    weights = torch.linspace(-0.7, 1.3, features.numel(), device=features.device).reshape_as(features)
    loss = (features * weights).sum() + 0.03 * state.square().sum()
    loss.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in net.named_parameters()}
    return features.detach(), state.detach(), gradients


@pytest.mark.parametrize("kind", ["policy", "value"])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_cpu_adapter_preserves_outputs_and_parameter_gradients(make_system, kind, sequence, fused):
    _, policy, value = make_system("tabular_mdp")
    original = policy if kind == "policy" else value
    adapted = deepcopy(original)
    state_keys = list(original.state_dict())
    expected = _forward_backward(original, _inputs(sequence=sequence), kind)
    with device_adapters(fused_layer_norm=fused, precompute_inputs=True):
        actual = _forward_backward(adapted, _inputs(sequence=sequence), kind)
        assert list(adapted.state_dict()) == state_keys
        initial_state = adapted.initial_state(3)
        assert initial_state.device == next(adapted.parameters()).device
        assert initial_state.dtype == next(adapted.parameters()).dtype
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=3e-6, rtol=2e-5)
    assert actual[2].keys() == expected[2].keys()
    for name in actual[2]:
        torch.testing.assert_close(actual[2][name], expected[2][name], atol=1e-5, rtol=1e-4,
                                   msg=lambda message: f"{name}: {message}")


def test_context_restores_original_callables_after_exception():
    originals = tabular.one_hot, GRU.initial_state, LayerNorm.forward, GRU.forward
    with pytest.raises(RuntimeError, match="intentional"):
        with device_adapters():
            assert tabular.one_hot is not originals[0]
            assert GRU.initial_state is not originals[1]
            assert LayerNorm.forward is not originals[2]
            assert GRU.forward is not originals[3]
            raise RuntimeError("intentional")
    assert (tabular.one_hot, GRU.initial_state, LayerNorm.forward, GRU.forward) == originals


def test_initial_state_follows_parameter_dtype():
    gru = GRU(input_dim=5, hidden_dim=8).double()
    with device_adapters():
        state = gru.initial_state(4)
    assert state.dtype == torch.float64
    assert state.shape == (4, 8)
    assert torch.count_nonzero(state) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime not installed in CPU environment")
@pytest.mark.parametrize("kind", ["policy", "value"])
def test_cuda_adapter_matches_cpu_reference_forward_and_gradients(make_system, kind):
    _, policy, value = make_system("tabular_mdp")
    original = policy if kind == "policy" else value
    adapted = deepcopy(original).cuda()
    expected = _forward_backward(original, _inputs(), kind)
    # Disable TF32 for the tight numeric comparison; benchmarks may opt in explicitly.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with device_adapters():
            actual = _forward_backward(adapted, _inputs("cuda"), kind)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(actual[1].cpu(), expected[1], atol=2e-5, rtol=1e-4)
    for name in actual[2]:
        torch.testing.assert_close(actual[2][name].cpu(), expected[2][name], atol=1e-4, rtol=5e-4,
                                   msg=lambda message: f"{name}: {message}")
