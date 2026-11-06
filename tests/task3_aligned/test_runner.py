"""Small mocked orchestration tests; no engine, GPU or training is executed."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl2_reproduction.task3.sequence_batch import TrialRecord
from rl2_reproduction.task3_aligned.config import default_config, validate_config
from rl2_reproduction.task3_aligned import checkpoint, runner


def test_default_config_freezes_approved_budget_and_hyperparameters(tmp_path):
    config = default_config(tmp_path)
    validate_config(config)
    for section, key, value in [('run', 'max_training_steps', 5000001),
                                ('run', 'steps_per_update', 16000),
                                ('ppo', 'learning_rate', .001),
                                ('model', 'recurrent_hidden_dim', 128)]:
        altered = deepcopy(config)
        altered[section][key] = value
        with pytest.raises(ValueError, match='settings differ'):
            validate_config(altered)


@pytest.fixture
def mocked_run(monkeypatch, tmp_path):
    from rl2_reproduction.task3_aligned import model, learner, parallel, evaluation, reporting
    events = {'collections': [], 'evaluations': [], 'updates': 0,
              'fail_update_once': False, 'fail_collection_once': False}

    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return torch.device('cpu')

        @property
        def architecture(self):
            return {'synthetic_runner_fixture': True}

        def to(self, *args, **kwargs):
            return self

    def record(task_id):
        summaries = [{'episode_index': i, 'length': length, 'success': False,
                      'return': -.04*length, 'collision_count': 0, 'physical_distance': 0.,
                      'turn_count': length, 'turn_fraction': 1.}
                     for i, length in enumerate((249, 248))]
        return TrialRecord({'actions': np.zeros(497, dtype=np.int64)}, task_id, summaries)

    class Collector:
        def __init__(self, **kwargs):
            self.physical_actions = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_weights(self, state):
            pass

        def collect(self, tasks, target_steps, start_trial_index, progress=None):
            if events['fail_collection_once']:
                events['fail_collection_once'] = False
                self.physical_actions += 7
                raise RuntimeError('injected collection failure after seven physical actions')
            batch = [record(tasks[(start_trial_index+i) % len(tasks)]['task_id'])
                     for i in range((target_steps+496)//497)]
            actions = len(batch)*497
            self.physical_actions += actions
            events['collections'].append((target_steps, start_trial_index, actions))
            return batch, {'next_trial_index': start_trial_index+len(batch), 'env_actions': actions,
                           'physical_actions': actions, 'trial_count': len(batch),
                           'task_ids': [r.task_id for r in batch]}

    class Learner:
        def __init__(self, policy, *args, **kwargs):
            self.policy = policy

        def update(self, batch):
            events['updates'] += 1
            if events['fail_update_once']:
                events['fail_update_once'] = False
                raise RuntimeError('injected post-collection update failure')
            with torch.no_grad():
                self.policy.weight.add_(1.)
            return {'training_steps': sum(r.length for r in batch), 'optimizer_steps': 1}

    def evaluate(state, tasks, **kwargs):
        events['evaluations'].append((tasks[0]['split'], len(tasks), float(state['weight'][0])))
        return {'policy': {'episodes': [{'success_rate': 0.}, {'success_rate': 0.}]},
                'environment_steps': 10*len(tasks)}

    def manifest(path):
        split = Path(path).stem
        count = 200 if split == 'val' else 1000
        return [{'task_id': f'{split}-{i:04d}', 'split': split, 'canonical_group_id': str(i % 3)}
                for i in range(count)]

    # Production configuration validation is tested separately above. A tiny
    # test-only budget exercises the real loop and its final-trial overshoot.
    monkeypatch.setattr(runner, 'validate_config', lambda config: None)
    monkeypatch.setattr(runner, 'configure', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'data_identity', lambda config: {'synthetic_data': True})
    monkeypatch.setattr(runner, 'load_manifest', manifest)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'mock-only-no-cuda')
    monkeypatch.setattr(runner, 'capture_rng', lambda: {})
    monkeypatch.setattr(runner, 'restore_rng', lambda state: None)
    monkeypatch.setattr(checkpoint, 'capture_rng', lambda: {})
    monkeypatch.setattr(checkpoint, 'restore_rng', lambda state: None)
    monkeypatch.setattr(model, 'WeightNormVisualPolicy', Model)
    monkeypatch.setattr(learner, 'AlignedPPOLearner', Learner)
    monkeypatch.setattr(parallel, 'ParallelCollector', Collector)
    monkeypatch.setattr(evaluation, 'evaluate_parallel', evaluate)
    monkeypatch.setattr(reporting, 'generate_report', lambda *args, **kwargs: None)
    config = default_config(tmp_path)
    config['run']['max_training_steps'] = 1001
    config['run']['steps_per_update'] = 500
    return config, tmp_path/'run', events


def test_runner_stops_at_complete_trial_cap_and_only_final_test(mocked_run):
    config, output, events = mocked_run
    result = runner.run_experiment(config, output)
    assert result['training_steps'] == 1491 <= 1001+499
    assert result['updates'] == 2 and result['trials'] == 3
    assert events['collections'] == [(500, 0, 994), (7, 2, 497)]
    assert events['evaluations'] == [('val', 200, 0.), ('val', 200, 2.), ('test', 1000, 2.)]
    assert result['evaluation_environment_steps'] == 14000
    assert not (output/'active.lock').exists()
    assert json.loads((output/'status.json').read_text())['state'] == 'completed'
    with pytest.raises(FileExistsError, match='overwrite'):
        runner.run_experiment(config, output)


def test_failed_update_reuses_archived_trials_without_resampling(mocked_run):
    config, output, events = mocked_run
    events['fail_update_once'] = True
    with pytest.raises(RuntimeError, match='injected'):
        runner.run_experiment(config, output)
    assert (output/'rollouts/update_0001.npz').exists()
    assert (output/'initial.pt').exists() and not (output/'active.lock').exists()
    result = runner.run_experiment(config, output, resume=True)
    assert result['training_steps'] == 1491
    assert events['collections'] == [(500, 0, 994), (7, 2, 497)]
    assert events['updates'] == 3  # Failed attempt + the two successful updates.
    assert events['evaluations'] == [('val', 200, 0.), ('val', 200, 2.), ('test', 1000, 2.)]
    assert len((output/'metrics.jsonl').read_text().splitlines()) == 2


def test_resume_uses_immutable_checkpoint_and_repairs_evaluation_index(mocked_run):
    config, output, events = mocked_run
    runner.run_experiment(config, output)
    # Crash windows: a stale latest pointer, truncated metrics append, and a
    # final evaluation written before its index entry was committed.
    (output/'latest.pt').write_bytes((output/'initial.pt').read_bytes())
    (output/'metrics.jsonl').write_text('uncommitted partial row')
    index = json.loads((output/'evaluation_index.json').read_text())
    (output/'evaluation_index.json').write_text(json.dumps(index[:-1]))
    events_before = deepcopy(events)
    result = runner.run_experiment(config, output, resume=True)
    assert result['training_steps'] == 1491
    assert events == events_before
    assert len((output/'metrics.jsonl').read_text().splitlines()) == 2
    restored = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
    assert restored['update'] == 2
    assert len(json.loads((output/'evaluation_index.json').read_text())) == 3


def test_unarchived_collection_failure_refuses_to_repeat_physical_actions(mocked_run):
    config, output, events = mocked_run
    events['fail_collection_once'] = True
    with pytest.raises(RuntimeError, match='seven physical actions'):
        runner.run_experiment(config, output)
    inflight = json.loads((output/'collection_inflight.json').read_text())
    assert inflight['physical_actions'] == 7 and inflight['failed']
    with pytest.raises(RuntimeError, match='refusing to repeat physical actions'):
        runner.run_experiment(config, output, resume=True)
    assert events['updates'] == 0 and events['collections'] == []
    assert not (output/'active.lock').exists()
