"""Synthetic checkpoint acceptance fixtures; no engine data required."""

from pathlib import Path

import argparse

import copy

import gc

import json

import os

import random

import threading

import time

import numpy as np

import torch

from rl2_reproduction.task3.sequence_batch import TrialRecord, collate_trials, compute_gae

from rl2_reproduction.task3.utils import atomic_json, capture_rng, configure, sha256

from rl2_reproduction.task3_aligned.model import WeightNormVisualPolicy

from rl2_reproduction.task3_aligned.learner import AlignedPPOLearner, compute_losses, replay, _pad_graph_batch, _CudaGraphLoss, _CudaGraphForward

from rl2_reproduction.task3_aligned.checkpoint import save_checkpoint, load_checkpoint

from rl2_reproduction.task3_aligned.config import PPO

def _optimizer(policy):
    return torch.optim.AdamW(policy.parameters(), lr=PPO['learning_rate'], eps=PPO['adam_eps'], weight_decay=PPO['weight_decay'])

def synthetic_trials(policy, lengths=(18, 11), seed=81):
    """Complete two-episode protocol fixtures; these are NOT engine rollouts."""
    rng = np.random.default_rng(seed)
    records = []
    for index, length in enumerate(lengths):
        assert 2 <= length <= 500
        first = length // 2
        actions = rng.integers(0, 3, length, dtype=np.int64)
        rewards = np.full(length, -.04, dtype=np.float32)
        rewards[[first-1, length-1]] += 1
        done = np.zeros(length, dtype=np.bool_); done[[first-1, length-1]] = True
        trial_done = np.zeros_like(done); trial_done[-1] = True
        resets = np.zeros_like(done); resets[0] = True
        arrays = dict(images_uint8=rng.integers(0, 256, (length,3,30,40), dtype=np.uint8), actions=actions,
                      rewards=rewards, episode_done=done, trial_done=trial_done, reset_mask=resets,
                      prev_actions=np.r_[np.int64(0), actions[:-1]], prev_rewards=np.r_[np.float32(0), rewards[:-1]],
                      prev_episode_done=np.r_[np.float32(0), done[:-1].astype(np.float32)],
                      episode_index=np.r_[np.zeros(first,dtype=np.int64),np.ones(length-first,dtype=np.int64)],
                      old_logits=np.zeros((length,3),dtype=np.float32), old_values=np.zeros(length,dtype=np.float32),
                      old_log_probs=np.zeros(length,dtype=np.float32))
        summaries=[dict(episode_index=i,length=b-a,success=True,return_=float(rewards[a:b].sum()))
                   for i,(a,b) in enumerate(((0,first),(first,length)))]
        records.append(TrialRecord(arrays,f'synthetic-{seed}-{index}',summaries))
    batch=collate_trials(records,policy.device)
    with torch.no_grad():
        logits,values,_=replay(policy,batch)
        logp=torch.distributions.Categorical(logits=logits).log_prob(batch['actions'])
    for i,record in enumerate(records):
        for name,value in (('old_logits',logits),('old_values',values),('old_log_probs',logp)):
            record.arrays[name]=value[i,:record.length].cpu().numpy().copy()
    return records

def _compare(a,b,*,atol,rtol,label):
    """Compare ALL tensor entries, retaining maxima rather than only a pass flag."""
    rows=[]
    def visit(x,y,path):
        if isinstance(x,torch.Tensor):
            x,y=x.detach().cpu(),y.detach().cpu()
            torch.testing.assert_close(x,y,atol=atol,rtol=rtol,msg=lambda m:f'{label}/{path}: {m}')
            if x.numel() and x.is_floating_point():
                delta=(x-y).abs()
                scale=atol+rtol*y.abs()
                scaled=torch.where(scale>0,delta/scale.clamp_min(torch.finfo(delta.dtype).tiny),torch.zeros_like(delta))
                rows.append(dict(name=path,max_abs=float(delta.max()),max_scaled=float(scaled.max())))
        elif isinstance(x,np.ndarray):
            np.testing.assert_array_equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),f'{label}/{path} keys'
            for key in x: visit(x[key],y[key],f'{path}.{key}')
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for i,(left,right) in enumerate(zip(x,y)):visit(left,right,f'{path}[{i}]')
        else:
            assert x==y,f'{label}/{path}: {x!r} != {y!r}'
    visit(a,b,'root')
    return dict(label=label,atol=atol,rtol=rtol,tensors=len(rows),max_abs=max((r['max_abs'] for r in rows),default=0.),
                worst_scaled=max(rows,key=lambda r:r['max_scaled'],default=None))

def checkpoint_resume(output,device='cuda',use_cuda_graph=True):
    configure(831,1)
    policy=WeightNormVisualPolicy(device=device);optimizer=_optimizer(policy)
    shuffle=torch.Generator().manual_seed(831);config={'ppo':dict(PPO,epochs=1,minibatch_trials=2)}
    learner=AlignedPPOLearner(policy,optimizer,config,shuffle,use_cuda_graph=use_cuda_graph)
    first=synthetic_trials(policy,(18,11,7),seed=71)
    learner.update(first)
    path=output/'resume_acceptance.pt'
    save_checkpoint(path,policy,optimizer,config,{'fixture':'synthetic'},update=1,training_steps=sum(t.length for t in first),
                    next_trial_index=3,shuffle_rng=shuffle,extra={'validation_only':True})
    second=synthetic_trials(policy,(21,13,6),seed=72)
    learner.update(second)
    expected_model=copy.deepcopy(policy.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict())
    expected_step_devices=[str(state['step'].device) for state in optimizer.state.values()]
    expected_shuffle=shuffle.get_state().clone()
    expected_random=(random.random(),float(np.random.rand()),torch.rand(4),torch.rand(4,device=device))
    del learner,policy,optimizer
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    resumed=WeightNormVisualPolicy(device=device);opt=_optimizer(resumed);sh=torch.Generator()
    state=load_checkpoint(path,resumed,opt,config=config,hashes={'fixture':'synthetic'},shuffle_rng=sh,restore_random=True)
    assert state['update']==1 and state['next_trial_index']==3
    continuation=AlignedPPOLearner(resumed,opt,config,sh,use_cuda_graph=use_cuda_graph)
    continuation.update(second)
    actual_step_devices=[str(state['step'].device) for state in opt.state.values()]
    assert expected_step_devices==actual_step_devices
    assert set(actual_step_devices)=={'cpu'},'Non-capturable AdamW step counters must stay on CPU'
    actual_random=(random.random(),float(np.random.rand()),torch.rand(4),torch.rand(4,device=device))
    checks=[_compare(expected_model,resumed.state_dict(),atol=0,rtol=0,label='same-device-resume/model'),
            _compare(expected_optimizer,opt.state_dict(),atol=0,rtol=0,label='same-device-resume/AdamW'),
            _compare(expected_shuffle,sh.get_state(),atol=0,rtol=0,label='same-device-resume/shuffle-RNG'),
            _compare(expected_random,actual_random,atol=0,rtol=0,label='same-device-resume/global-RNG')]
    rejected=[]
    for kwargs in ({'config':{'changed':True}},{'hashes':{'changed':True}}):
        try:load_checkpoint(path,resumed,**kwargs)
        except ValueError:rejected.append(next(iter(kwargs)))
        else:raise AssertionError('Expected provenance rejection')
    result=dict(passed=True,device=device,cuda_graph=use_cuda_graph,checks=checks,rejected_mismatch=rejected,
                optimizer_step_devices_identical=True,optimizer_step_device='cpu',
                checkpoint_sha256=sha256(path),fixture='synthetic short complete-trial updates')
    atomic_json(output/f'resume_{device}.json',result)
    return result
