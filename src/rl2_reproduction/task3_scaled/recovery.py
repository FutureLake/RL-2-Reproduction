"""Explicit audited migration for lost collection; never silently erase actions."""
import json
from rl2_reproduction.task3.utils import sha256


def read_recovery(output, hashes):
    path = output / 'recovery.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['new_hashes'] != hashes:
        raise ValueError('Recovery current code/data provenance differs')
    if {k:v for k,v in record['old_hashes'].items() if k != 'sources'} != {
            k:v for k,v in hashes.items() if k != 'sources'}:
        raise ValueError('Recovery cannot change data identity')
    old_sources, new_sources = record['old_hashes'].get('sources', {}), hashes.get('sources', {})
    changed = {k for k in old_sources.keys() | new_sources.keys() if old_sources.get(k) != new_sources.get(k)}
    if not changed <= {'task3_scaled/runner.py', 'task3_scaled/checkpoint.py',
                       'task3_scaled/atomic_io.py', 'task3_scaled/recovery.py'}:
        raise ValueError('Recovery source migration exceeds the IO/accounting-only allowlist')
    if type(record['discarded_training_actions']) is not int or record['discarded_training_actions'] <= 0:
        raise ValueError('Recovery requires a positive integer physical-action charge')
    if type(record['inflight_action_reserve']) is not int or record['inflight_action_reserve'] != 4:
        raise ValueError('Recovery reserves one unreported in-flight action per CPU worker')
    checkpoint = output / record['checkpoint_file']
    if sha256(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Recovery source checkpoint changed')
    evidence = output / record['inflight_evidence_file']
    if sha256(evidence) != record['inflight_sha256']:
        raise ValueError('Recovery inflight evidence changed')
    inflight = json.loads(evidence.read_text())
    if (inflight['physical_actions'] != record['discarded_training_actions']
            or inflight['committed_training_actions'] != record['training_steps']
            or inflight['start_trial_index'] != record['next_trial_index']
            or inflight['update'] != record['update'] + 1 or not inflight['failed']):
        raise ValueError('Recovery counters disagree with failure evidence')
    return record


def checkpoint_hashes(path, current_hashes, recovery):
    if recovery and sha256(path) == recovery['checkpoint_sha256']:
        return recovery['old_hashes']
    return current_hashes


def reconcile_inflight(path, recovery, updates, steps, trial_index):
    """Only acknowledge the exact archived failure explicitly authorized for recovery."""
    if not recovery or not path.exists():
        return False
    if sha256(path) != recovery['inflight_sha256']:
        return False
    if (updates,steps,trial_index) != (recovery['update'],recovery['training_steps'],recovery['next_trial_index']):
        raise ValueError('Recovery can acknowledge inflight only at its source checkpoint')
    path.unlink()
    return True
