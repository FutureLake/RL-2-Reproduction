"""Windows-safe publication of JSON without changing experiment RNG state.

Durable metadata fails loudly after bounded retries. The status snapshot is
telemetry only: a locked destination may skip one publication and recover on
the next progress event. Neither path removes the existing destination first.
"""
from pathlib import Path
import json
import os
import sys
import tempfile
import time


_last_status_warning = {}


def is_transient_lock(error):
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33)


def replace_with_retry(source, destination, *, timeout=2.0, initial_delay=0.02, max_delay=0.2):
    """Atomically replace destination; retry only permission/sharing failures.

The source remains owned by the caller if retries are exhausted. No random
backoff is used, and the previous destination remains intact on failure.
"""
    if timeout < 0 or initial_delay <= 0 or max_delay <= 0:
        raise ValueError("Retry timeout must be nonnegative and delays positive")
    deadline = time.monotonic() + timeout
    delay = min(initial_delay, max_delay)
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            remaining = deadline - time.monotonic()
            if not is_transient_lock(error) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(max_delay, delay * 2)


def atomic_json(path, value, *, timeout=2.0):
    """Write, flush, fsync, then atomically publish a same-directory temp file."""
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temp, path, timeout=timeout)
    finally:
        temp.unlink(missing_ok=True)


def write_status(path, row, *, timeout=0.5, warning_interval=30.0):
    """Return publication success; only exhausted file locks are nonfatal.

The caller can keep printing the full progress row to stdout regardless of
this return value. Corrupt data, disk-full errors and unrelated OS errors
still propagate; they are not classified as harmless reader contention.
"""
    try:
        atomic_json(path, row, timeout=timeout)
        return True
    except OSError as error:
        if not is_transient_lock(error):
            raise
        key = str(Path(path).resolve())
        now = time.monotonic()
        if now - _last_status_warning.get(key, float("-inf")) >= warning_interval:
            print(f"Status publication skipped after bounded lock retries: {path}; "
                  f"state={row.get('state')}, update={row.get('update')}; {error}",
                  file=sys.stderr, flush=True)
            _last_status_warning[key] = now
        return False
