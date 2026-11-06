"""Synthetic checkpoint integration: no policy evaluation or ViZDoom actions."""
from pathlib import Path
import importlib.util
import pytest
import torch


def _acceptance_module():
    path=Path(__file__).resolve().parents[2]/'scripts/validate_task3_aligned.py'
    spec=importlib.util.spec_from_file_location('validate_task3_aligned',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_complete_update_resume_model_optimizer_and_all_rng(tmp_path):
    result=_acceptance_module().checkpoint_resume(tmp_path,device='cpu',use_cuda_graph=False)
    assert result['passed']
    assert result['rejected_mismatch']==['config','hashes']
    assert all(row['max_abs']==0 for row in result['checks'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware / build unavailable')
def test_cuda_graph_complete_update_resume_model_optimizer_and_all_rng(tmp_path):
    result=_acceptance_module().checkpoint_resume(tmp_path,device='cuda',use_cuda_graph=True)
    assert result['passed']
    assert all(row['max_abs']==0 for row in result['checks'])
