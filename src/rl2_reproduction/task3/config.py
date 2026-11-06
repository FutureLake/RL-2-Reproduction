"""Reject settings the v2 implementation would otherwise silently ignore."""
def validate_config(config):
    fixed = {
        "env": {"engine": "vizdoom", "mode": "synchronous", "maze_size": 5, "cell_size_doom": 128,
                "forward_distance_cells": .2, "turn_degrees": 5, "ticks_per_action": 4,
                "episodes_per_trial": 2, "max_steps_per_episode": 250, "same_start_pose_within_trial": True,
                "reward_goal": 1., "reward_step": -.04, "reward_collision": -.001},
        "observation": {"width": 40, "height": 30, "channels": 3, "resize": "area",
                        "normalization": "minus_one_to_one", "privileged_inputs": False},
        "model": {"conv_channels": [16, 16], "conv_kernels": [5, 5], "conv_strides": [2, 2],
                  "conv_padding": 0, "action_embedding_dim": 256, "fusion_dim": 256,
                  "recurrent_cell": "upstream_relu_layernorm_gru", "recurrent_hidden_dim": 256,
                  "shared_actor_critic": True, "pretrained_weights": None, "train_visual_encoder": True},
        "ppo": {"optimizer": "adamw", "value_loss": "huber", "standardize_advantages": False, "full_trial_bptt": True},
    }
    for section, fields in fixed.items():
        for key, expected in fields.items():
            if config.get(section, {}).get(key, object()) != expected:
                raise ValueError(f"Unsupported v2 setting {section}.{key}; expected {expected!r}")
    run = config["run"]
    if run["profile"] not in ("smoke", "pilot") or run["num_envs"] != 1 or run.get("mode", "rl2") not in ("rl2", "b1"):
        raise ValueError("Only smoke/pilot, single process and rl2/b1 are supported")
    if run["profile"] == "smoke":
        settings = {"updates": 5, "trials_per_update": 8, "max_training_steps": 20000, "device": "cpu", "seed": 0, "mode": "rl2"}
        if any(run.get(k) != v for k, v in settings.items()) or config["ppo"]["epochs"] != 2 or config["ppo"]["minibatch_trials"] != 4:
            raise ValueError("S4 smoke fixes CPU/seed0/full model/5x8 trials/2 epochs/minibatch4/cap20000")
    elif run["max_training_steps"] != 500000 or run["steps_per_update"] != 16000:
        raise ValueError("Pilot budget is 500000 nominal steps, 16000 steps/update, finishing last trial only")
    if "evaluation" in config or "visual_version" in config:
        from .pilot_protocol import EVALUATION, VISUAL_VERSION
        if run["profile"] != "pilot" or config.get("evaluation") != EVALUATION or config.get("visual_version") != VISUAL_VERSION:
            raise ValueError("S5 stone-goal-v2 has a fixed evaluation schedule and visual version")
        if any(run.get(k) != v for k, v in {"device": "cpu", "seed": 0, "mode": "rl2", "resume": False}.items()):
            raise ValueError("S5 is CPU seed0 RL2 from random initialization, no implicit resume")
