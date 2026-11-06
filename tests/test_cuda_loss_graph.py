"""CUDA graph PPO must agree with eager forward/backward and AdamW updates."""

import copy

import pytest
import torch
from torch.distributions import Distribution

from rl2_reproduction.benchmark import compute_losses
from rl2_reproduction.cuda_loss_graph import CudaLossGraph
from rl2_reproduction.device_adapter import device_adapters


def _batch(seed):
    generator = torch.Generator().manual_seed(seed)
    shape = (4, 6)
    return {"obs": torch.randint(0, 4, shape, generator=generator).cuda(),
            "acs": torch.randint(0, 3, shape, generator=generator).cuda(),
            "rews": torch.randn(shape, generator=generator).cuda(),
            "dones": torch.randint(0, 2, shape, generator=generator).float().cuda(),
            "logpacs": (-1.1 + 0.2 * torch.randn(shape, generator=generator)).cuda(),
            "vpreds": torch.randn(shape, generator=generator).cuda(),
            "advs": torch.linspace(-1.5, 1.7, 24).reshape(shape).cuda(),
            "tdlam_rets": torch.randn(shape, generator=generator).cuda()}


def test_cuda_graph_rejects_cpu_models(make_system):
    _, policy, value = make_system()
    with pytest.raises(ValueError, match="CUDA"):
        CudaLossGraph(compute_losses, policy, value, {"obs": torch.zeros(4, 6)}, 0.1, 0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_graph_matches_eager_losses_gradients_and_two_adam_steps(make_system, environment):
    _, policy, value = make_system(environment)
    eager_nets = [policy.cuda(), value.cuda()]
    graph_nets = copy.deepcopy(eager_nets)
    eager_optimizers = [torch.optim.AdamW(net.parameters(), lr=2e-4, eps=1e-5) for net in eager_nets]
    graph_optimizers = [torch.optim.AdamW(net.parameters(), lr=2e-4, eps=1e-5) for net in graph_nets]
    before = [{key: tensor.clone() for key, tensor in net.state_dict().items()} for net in graph_nets]
    rng_before = torch.cuda.get_rng_state_all()
    old_validation = Distribution._validate_args
    with device_adapters():
        graph = CudaLossGraph(compute_losses, *graph_nets, _batch(211), 0.1, 0.01)
        assert Distribution._validate_args == old_validation
        assert all(torch.equal(left, right) for left, right in zip(rng_before, torch.cuda.get_rng_state_all()))
        for net, initial in zip(graph_nets, before):
            assert all(torch.equal(tensor, initial[key]) for key, tensor in net.state_dict().items())
            assert all(torch.count_nonzero(p.grad) == 0 for p in net.parameters())
        for seed in (211, 212):
            data = _batch(seed)
            for optimizer in eager_optimizers:
                optimizer.zero_grad(set_to_none=True)
            eager_losses = compute_losses(data, *eager_nets, 0.1, 0.01)
            eager_losses["policy_loss"].backward()
            eager_losses["value_loss"].backward()
            graph_losses = graph.run(data)
            assert all(not loss.requires_grad for loss in graph_losses.values())
            for key in eager_losses:
                torch.testing.assert_close(graph_losses[key], eager_losses[key], atol=2e-6, rtol=2e-5)
            for expected, actual in zip(eager_nets, graph_nets):
                for left, right in zip(expected.parameters(), actual.parameters()):
                    assert left.grad is not None and right.grad is not None
                    torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=1e-4)
            for optimizer in eager_optimizers + graph_optimizers:
                optimizer.step()
            for expected, actual in zip(eager_nets, graph_nets):
                for left, right in zip(expected.parameters(), actual.parameters()):
                    torch.testing.assert_close(right, left, atol=2e-6, rtol=2e-5)
            for expected, actual in zip(eager_optimizers, graph_optimizers):
                left, right = expected.state_dict()["state"], actual.state_dict()["state"]
                for parameter_id in left:
                    for field in left[parameter_id]:
                        torch.testing.assert_close(right[parameter_id][field], left[parameter_id][field], atol=2e-6, rtol=1e-4)
        wrong_shape = {key: tensor[:2] for key, tensor in _batch(213).items()}
        with pytest.raises(ValueError, match="shape"):
            graph.run(wrong_shape)
