"""Write an explicit local 5M or 20M config, pinning regenerated dataset hashes."""
import argparse
import importlib
import json
from pathlib import Path
import yaml
from rl2_reproduction.task3.utils import PROJECT, sha256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("5m", "20m"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path)
    args = parser.parse_args()
    module = importlib.import_module("rl2_reproduction." + ("task3_aligned" if args.profile == "5m" else "task3_scaled") + ".config")
    config = module.default_config(PROJECT)
    data = Path(config["data_root"])
    config["manifest_hashes"] = {s: sha256(data / f"manifests/{s}.jsonl") for s in ("train", "val", "test")}
    if args.profile == "20m":
        if args.parent_run is None:
            parser.error("--parent-run must point to your completed 5M run")
        config["parent_run"] = str(args.parent_run.resolve())
        config["parent_checkpoint_sha256"] = sha256(args.parent_run / "latest.pt")
    module.validate_config(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(json.dumps(config, indent=2))
