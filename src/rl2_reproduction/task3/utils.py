"""Small provenance and RNG tools without legacy MPI trainer imports."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
UPSTREAM_SHA = "51afa4b6c6d05b469c7f74076b9490ae0809b289"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def namespace_seed(seed, namespace):
    return int(np.random.SeedSequence([seed, namespace]).generate_state(1, dtype=np.uint64)[0])


def configure(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(namespace_seed(seed, 10))
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])


def provenance():
    upstream = PROJECT / "third_party" / "pytorch_rl2"
    sha = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if sha != UPSTREAM_SHA or dirty:
        raise RuntimeError("Pinned upstream changed")
    names = ["numpy", "torch", "vizdoom", "Pillow", "matplotlib", "PyYAML", "pytest"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"platform": platform.platform(), "python": platform.python_version(),
            "packages": versions, "upstream_commit": sha, "upstream_clean": True,
            "numpy_build_dependencies": np.show_config(mode="dicts").get("Build Dependencies", {}),
            "task3_source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
