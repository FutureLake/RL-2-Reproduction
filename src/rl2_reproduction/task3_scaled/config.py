"""Explicit experiment settings; reject silent changes to the approved protocol."""
from pathlib import Path
from . import PROTOCOL

PPO = {"gamma": .99, "gae_lambda": .99, "optimizer": "adamw", "learning_rate": .0003,
       "adam_eps": 1e-5, "weight_decay": 0., "clip_epsilon": .2, "value_loss": "huber",
       "value_coef": .5, "entropy_coef": .01, "max_grad_norm": .5,
       "standardize_advantages": False, "epochs": 4, "minibatch_trials": 8,
       "full_trial_bptt": True, "epoch_stop_kl": .03}
ENV = {"engine": "vizdoom", "mode": "synchronous", "maze_size": 5, "cell_size_doom": 128,
       "forward_distance_cells": .2, "turn_degrees": 5, "ticks_per_action": 4,
       "episodes_per_trial": 2, "max_steps_per_episode": 250, "same_start_pose_within_trial": True,
       "reward_goal": 1., "reward_step": -.04, "reward_collision": -.001}
OBSERVATION = {"width": 40, "height": 30, "channels": 3, "resize": "area",
               "normalization": "minus_one_to_one", "privileged_inputs": False}
MODEL = {"conv_channels": [16, 16], "conv_kernels": [5, 5], "conv_strides": [2, 2],
         "conv_padding": 0, "action_embedding_dim": 256, "fusion_dim": 256,
         "recurrent_cell": "weightnorm_relu_gru", "recurrent_hidden_dim": 256,
         "shared_actor_critic": True, "pretrained_weights": None, "train_visual_encoder": True}
EVALUATION = {"initial_tasks": 200, "final_tasks": 200, "test_tasks": 1000,
              "every_updates": 5, "subset_per_geometry_group": 10,
              "milestone_training_steps": [5000000,10000000,15000000], "initial_mode": "rl2",
              "intermediate_mode": "rl2", "milestone_mode": "paired",
              "final_mode": "paired", "test_mode": "paired",
              "action_seed": 10001, "trace_per_geometry_group": 1,
              "test_checkpoint": "final", "selection_uses_test": False}


def validate_config(config):
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Wrong aligned protocol")
    for key, expected in (("ppo", PPO), ("env", ENV), ("observation", OBSERVATION),
                          ("model", MODEL), ("evaluation", EVALUATION)):
        if config.get(key) != expected:
            raise ValueError(f"Approved {key} settings differ")
    run = config["run"]
    expected = {"profile": "scaled_20m", "device": "cuda", "actor_device": "cpu", "workers": run['workers'],
                "actor_torch_threads": 1, "learner_torch_threads": 1, "seed": 0,
                "mode": "rl2", "max_training_steps": 20000000, "steps_per_update": 50000,
                "cuda_graph": True, "resume": False, "automatic_extension": False}
    if run != expected:
        raise ValueError("Approved 5M/50k/seed0/4CPU+CUDA run settings differ")
    if run['workers'] not in (4,6,8) or config.get('lr_schedule') != SCHEDULE:
        raise ValueError('Unapproved workers or learning-rate schedule')
    if not Path(config['parent_run']).is_absolute() or len(config['parent_checkpoint_sha256']) != 64:
        raise ValueError('Explicit parent checkpoint provenance required')
    if config.get("visual_version") != "rl2-task3-visual-stone-goal-v2":
        raise ValueError("Wrong frozen visual version")
    for key in ("project_root", "data_root"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an explicit absolute path")


def default_config(project_root):
    project_root = Path(project_root).resolve()
    return {"protocol": PROTOCOL, "project_root": str(project_root),
            "data_root": str(project_root / "data/task3/rl2-task3-visual-stone-goal-v2"),
            "visual_version": "rl2-task3-visual-stone-goal-v2",
            "env": dict(ENV), "observation": dict(OBSERVATION), "model": dict(MODEL),
            "ppo": dict(PPO), "evaluation": dict(EVALUATION),
            "lr_schedule":dict(SCHEDULE),
            "run": {"profile": "scaled_20m", "device": "cuda", "actor_device": "cpu", "workers": 4,
                    "actor_torch_threads": 1, "learner_torch_threads": 1, "seed": 0, "mode": "rl2",
                    "max_training_steps": 20000000, "steps_per_update": 50000, "cuda_graph": True,
                    "resume": False, "automatic_extension": False}}


SCHEDULE = {'kind':'warmup_hold_cosine','start_steps':5000498,'warmup_end':5500498,
            'decay_start':10000000,'end_steps':20000000,
            'start_lr':3e-4,'peak_lr':6e-4,'end_lr':3e-5}

def learning_rate_at(steps, schedule=None):
    import math
    s=SCHEDULE if schedule is None else schedule
    if steps <= s['start_steps']:return s['start_lr']
    if steps < s['warmup_end']:
        fraction=(steps-s['start_steps'])/(s['warmup_end']-s['start_steps'])
        return s['start_lr']+fraction*(s['peak_lr']-s['start_lr'])
    if steps <= s['decay_start']:return s['peak_lr']
    fraction=min(1.,(steps-s['decay_start'])/(s['end_steps']-s['decay_start']))
    return s['end_lr']+.5*(s['peak_lr']-s['end_lr'])*(1+math.cos(math.pi*fraction))
