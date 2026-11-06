"""Explicit new experiment lineage; parent artifacts remain read-only."""
from pathlib import Path
import json, shutil, platform, importlib.metadata
import yaml
from rl2_reproduction.task3.utils import atomic_json, sha256
from rl2_reproduction.task3.pilot_protocol import model_digest
from rl2_reproduction.task3_aligned.config import validate_config as validate_parent
from .checkpoint import load_checkpoint, save_checkpoint


def initialize_branch(config, output, policy, optimizer, shuffle_rng, hashes):
    parent=Path(config['parent_run']); checkpoint=parent/'latest.pt'
    if sha256(checkpoint)!=config['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint hash differs')
    old_config=json.loads((parent/'config.json').read_text());validate_parent(old_config)
    old_hashes=json.loads((parent/'manifest.json').read_text())['hashes']
    for key in ('data_root','visual_version','manifests','visual_config','calibration'):
        if old_hashes[key]!=hashes[key]:raise ValueError('Parent data identity differs: '+key)
    state=load_checkpoint(checkpoint,policy,optimizer,config=old_config,hashes=old_hashes,
                          shuffle_rng=shuffle_rng,restore_random=True)
    if not (100 == state['update'] and 5000000 <= state['training_steps'] <= 5000499):
        raise ValueError('Expected completed 5M seed0 parent boundary')
    lineage={'parent_run':str(parent),'parent_checkpoint_sha256':sha256(checkpoint),
             'parent_model_sha256':model_digest(policy),'parent_update':state['update'],
             'parent_training_steps':state['training_steps'],'parent_trials':state['next_trial_index'],
             'model_optimizer_rng_loaded':True,'new_training_actions':0,
             'prior_test_already_observed':True,
             'interpretation':'Continuation with changed LR schedule; final test is descriptive, not a fresh blind holdout.'}
    atomic_json(output/'lineage.json',lineage)
    atomic_json(output/'config.json',config)
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    atomic_json(output/'architecture.json',policy.architecture)
    atomic_json(output/'manifest.json',{'hashes':hashes,'packages':{n:importlib.metadata.version(n) for n in
                ('torch','numpy','vizdoom','Pillow','matplotlib','PyYAML')},'python':platform.python_version(),
                'initialization':'explicit continuation from completed seed0 5M checkpoint',**lineage})
    shutil.copyfile(parent/'metrics.jsonl',output/'parent_metrics.jsonl')
    history=[]
    for entry in json.loads((parent/'evaluation_index.json').read_text()):
        if not entry['file'].startswith('validation'):continue
        new_name='validation_milestone_5000000.json' if entry['file']=='validation_final.json' else entry['file']
        shutil.copyfile(parent/entry['file'],output/new_name)
        history.append({**entry,'file':new_name})
    atomic_json(output/'evaluation_index.json',history)
    save_checkpoint(output/'initial.pt',policy,optimizer,config,hashes,update=state['update'],
                    training_steps=state['training_steps'],next_trial_index=state['next_trial_index'],
                    shuffle_rng=shuffle_rng,extra=state['extra'])
    atomic_json(output/'startup.json',{'data':hashes,'architecture':policy.architecture,'seed':0,
                'device':str(policy.device),'workers':config['run']['workers'],
                'nominal_training_steps':20000000,'absolute_training_action_cap':20000499,
                'lr_schedule':config['lr_schedule'],**lineage})
    print(json.dumps({'startup':json.loads((output/'startup.json').read_text())}),flush=True)
