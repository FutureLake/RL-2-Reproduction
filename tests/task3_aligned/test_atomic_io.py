"""Windows sharing-lock regression: real handles plus non-lock failure cases."""
import ctypes
import errno
import json
import os
import random
import threading
import time

import pytest

from rl2_reproduction.task3_scaled import atomic_io


def _deny_delete(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # Share read/write, deliberately exclude FILE_SHARE_DELETE.
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing semantics")
def test_real_windows_reader_lock_retries_then_publishes(tmp_path):
    path = tmp_path / "status.json"
    atomic_io.atomic_json(path, {"old": True})
    close = _deny_delete(path)
    timer = threading.Timer(.12, close)
    timer.start()
    try:
        start = time.monotonic()
        atomic_io.atomic_json(path, {"new": "真实数据"}, timeout=1)
        assert time.monotonic() - start >= .10
        assert json.loads(path.read_text(encoding="utf-8")) == {"new": "真实数据"}
        assert list(tmp_path.glob("*.tmp")) == []
    finally:
        timer.join()


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing semantics")
def test_permanent_lock_durable_fails_status_skips_and_next_event_recovers(tmp_path, capsys):
    path = tmp_path / "status.json"
    atomic_io.atomic_json(path, {"old": True})
    close = _deny_delete(path)
    try:
        start = time.monotonic()
        with pytest.raises(PermissionError):
            atomic_io.atomic_json(path, {"new": True}, timeout=.08)
        assert time.monotonic() - start < 1
        assert not atomic_io.write_status(path, {"state": "training_collection", "update": 307}, timeout=.08)
        assert "Status publication skipped" in capsys.readouterr().err
        assert json.loads(path.read_text()) == {"old": True}
        assert list(tmp_path.glob("*.tmp")) == []
    finally:
        close()
    assert atomic_io.write_status(path, {"state": "ppo_update", "update": 307})
    assert json.loads(path.read_text())["state"] == "ppo_update"


@pytest.mark.parametrize("writer", [atomic_io.atomic_json, atomic_io.write_status])
def test_non_lock_io_failure_not_swallowed_and_old_data_survives(tmp_path, monkeypatch, writer):
    path = tmp_path / "status.json"
    path.write_text('{"old":true}')
    calls = []
    def fail(*args):
        calls.append(args)
        raise OSError(errno.ENOSPC, "Synthetic disk full")
    monkeypatch.setattr(atomic_io.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        writer(path, {"new": True})
    assert len(calls) == 1
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_writes_preserve_rng_and_reject_nan(tmp_path):
    import numpy as np
    import torch
    py = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    path = tmp_path / "status.json"
    atomic_io.atomic_json(path, {"valid": True})
    assert atomic_io.write_status(path, {"state": "test"})
    assert random.getstate() == py
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    with pytest.raises(ValueError):
        atomic_io.atomic_json(path, {"nan": float("nan")})
    assert json.loads(path.read_text()) == {"state": "test"}
    assert list(tmp_path.glob("*.tmp")) == []
