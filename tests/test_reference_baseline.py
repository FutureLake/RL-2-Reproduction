"""Behavioral checks for the original CPU GRU/PPO implementation.

These are pipeline and RL2-boundary tests, not claims of converged performance.
"""

import numpy as np
import pytest
import torch
from mpi4py import MPI

from rl2.algos.common import MetaEpisode, assign_credit, generate_meta_episode
from rl2.algos.ppo import compute_losses
from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import sync_grads


def sequence_inputs(meta_episode):
    """Reconstruct the causal inputs used in both rollout and PPO replay."""
    return {
        "curr_obs": torch.as_tensor(meta_episode.obs).long().unsqueeze(0),
        "prev_action": torch.as_tensor(np.r_[0, meta_episode.acs[:-1]]).long().unsqueeze(0),
        "prev_reward": torch.as_tensor(np.r_[0.0, meta_episode.rews[:-1]]).float().unsqueeze(0),
        "prev_done": torch.as_tensor(np.r_[1.0, meta_episode.dones[:-1]]).float().unsqueeze(0),
    }


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_gru_step_sequence_and_rollout_replay_agree(make_system, environment):
    env, policy, value = make_system(environment)
    meta = generate_meta_episode(env, policy, value, meta_episode_len=6)
    inputs = sequence_inputs(meta)
    with torch.no_grad():
        full_dist, full_policy_state = policy(**inputs, prev_state=policy.initial_state(1))
        full_values, full_value_state = value(**inputs, prev_state=value.initial_state(1))
        policy_state, value_state = policy.initial_state(1), value.initial_state(1)
        probs, values = [], []
        for t in range(6):
            step_inputs = {key: tensor[:, t] for key, tensor in inputs.items()}
            dist, policy_state = policy(**step_inputs, prev_state=policy_state)
            vpred, value_state = value(**step_inputs, prev_state=value_state)
            probs.append(dist.probs)
            values.append(vpred)

    assert full_dist.probs.shape == (1, 6, 3)
    assert full_values.shape == (1, 6)
    torch.testing.assert_close(full_dist.probs, torch.stack(probs, dim=1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(full_values, torch.stack(values, dim=1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(full_policy_state, policy_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(full_value_state, value_state, atol=2e-6, rtol=2e-6)
    replay_logprob = full_dist.log_prob(torch.as_tensor(meta.acs).unsqueeze(0))
    np.testing.assert_allclose(replay_logprob.numpy()[0], meta.logpacs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(full_values.numpy()[0], meta.vpreds, atol=2e-6, rtol=2e-6)


def test_sampler_retains_both_hidden_states_across_episodes_and_resets_trials(make_system):
    env, policy, value = make_system()
    records = {"policy": [], "value": []}

    def recorder(name):
        def hook(module, args, kwargs, output):
            records[name].append({
                **{key: tensor.detach().clone() for key, tensor in kwargs.items()},
                "new_state": output[1].detach().clone(),
            })
        return hook

    handles = [net.register_forward_hook(recorder(name), with_kwargs=True)
               for name, net in (("policy", policy), ("value", value))]
    try:
        trials = [generate_meta_episode(env, policy, value, 6) for _ in range(2)]
    finally:
        for handle in handles:
            handle.remove()

    for trial_index, trial in enumerate(trials):
        np.testing.assert_array_equal(trial.dones, [0, 1, 0, 1, 0, 1])
        for name in records:
            calls = records[name][trial_index * 6:(trial_index + 1) * 6]
            assert len(calls) == 6
            assert torch.count_nonzero(calls[0]["prev_state"]) == 0
            assert calls[0]["prev_action"].item() == 0
            assert calls[0]["prev_reward"].item() == 0
            assert calls[0]["prev_done"].item() == 1
            for t in range(1, 6):
                torch.testing.assert_close(calls[t]["prev_state"], calls[t - 1]["new_state"], rtol=0, atol=0)
                assert calls[t]["prev_action"].item() == trial.acs[t - 1]
                assert calls[t]["prev_reward"].item() == trial.rews[t - 1]
                assert calls[t]["prev_done"].item() == trial.dones[t - 1]
                assert calls[t]["curr_obs"].item() == trial.obs[t]
            # The env reset occurs before calls 2 and 4; recurrent memory survives.
            for t in (2, 4):
                assert calls[t]["prev_done"].item() == 1
                assert calls[t]["curr_obs"].item() == 0
                assert torch.count_nonzero(calls[t]["prev_state"]) > 0


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_seeded_sampling_is_reproducible(make_system, environment):
    trajectories = []
    for _ in range(2):
        env, policy, value = make_system(environment, seed=923)
        trajectories.append(generate_meta_episode(env, policy, value, 6))
    for field in ("obs", "acs", "rews", "dones", "logpacs", "vpreds"):
        np.testing.assert_array_equal(getattr(trajectories[0], field), getattr(trajectories[1], field))


def test_credit_crosses_episode_done_but_stops_at_trial_end():
    meta = MetaEpisode(num_timesteps=4, dummy_obs=0)
    meta.rews[:] = [1, 2, 3, 4]
    meta.vpreds[:] = [0.1, 0.2, 0.3, 0.4]
    meta.dones[:] = [0, 1, 0, 1]
    assign_credit(meta, gamma=1.0, lam=1.0)
    # With gamma=lambda=1, returns are MC sums over the WHOLE trial.
    np.testing.assert_allclose(meta.tdlam_rets, [10, 9, 7, 4], atol=1e-6)
    np.testing.assert_allclose(meta.advs, [9.9, 8.8, 6.7, 3.6], atol=1e-6)


def collect_batch(env, policy, value):
    return [assign_credit(generate_meta_episode(env, policy, value, 6), 0.99, 0.3)
            for _ in range(4)]


def update(batch, policy, value, policy_optimizer, value_optimizer):
    losses = compute_losses(batch, policy, value, clip_param=0.1, ent_coef=0.01)
    assert set(losses) == {"policy_loss", "value_loss", "meanent", "clipfrac"}
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    for name, net, optimizer in (("policy_loss", policy, policy_optimizer),
                                 ("value_loss", value, value_optimizer)):
        optimizer.zero_grad()
        losses[name].backward()
        grads = [p.grad for p in net.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        recurrent_grad = net._architecture._h2hhat.weight.grad
        assert recurrent_grad.abs().sum().item() > 0
        sync_grads(net, MPI.COMM_WORLD)
        optimizer.step()
    return losses


@pytest.mark.parametrize("environment", ["bandit", "tabular_mdp"])
def test_ppo_updates_gru_and_checkpoint_resumes_identically(make_system, environment, tmp_path):
    assert MPI.COMM_WORLD.Get_size() == 1, "Run this unit test suite without mpiexec."
    env, policy, value = make_system(environment)
    optimizers = [torch.optim.AdamW(net.parameters(), lr=2e-4) for net in (policy, value)]
    initial_gru = [net._architecture._h2hhat.weight.detach().clone() for net in (policy, value)]
    batch = collect_batch(env, policy, value)
    update(batch, policy, value, *optimizers)
    for net, before in zip((policy, value), initial_gru):
        assert not torch.equal(before, net._architecture._h2hhat.weight)

    for name, net, optimizer in zip(("policy", "value"), (policy, value), optimizers):
        save_checkpoint(1, str(tmp_path), name, net, optimizer, None)
    _, loaded_policy, loaded_value = make_system(environment, seed=19)
    loaded_optimizers = [torch.optim.AdamW(net.parameters(), lr=2e-4)
                         for net in (loaded_policy, loaded_value)]
    for name, original, loaded, optimizer in zip(
            ("policy", "value"), (policy, value), (loaded_policy, loaded_value), loaded_optimizers):
        step = maybe_load_checkpoint(str(tmp_path), name, loaded, optimizer, None, steps=None)
        assert step == 1
        for key in original.state_dict():
            torch.testing.assert_close(original.state_dict()[key], loaded.state_dict()[key], rtol=0, atol=0)
        assert optimizer.state_dict()["state"], "Adam moments must be restored."

    # Equality after the next update validates restoration of Adam state too.
    update(batch, policy, value, *optimizers)
    update(batch, loaded_policy, loaded_value, *loaded_optimizers)
    for original, loaded in ((policy, loaded_policy), (value, loaded_value)):
        for key in original.state_dict():
            torch.testing.assert_close(original.state_dict()[key], loaded.state_dict()[key], rtol=0, atol=0)
