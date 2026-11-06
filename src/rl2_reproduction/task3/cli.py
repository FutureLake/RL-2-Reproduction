"""Rebuildable Windows commands for Task 3 stages S0-S4."""
import argparse
import functools
import json
from pathlib import Path

from . import PROTOCOL
from .utils import PROJECT, atomic_json, sha256

DEFAULT_DATA = PROJECT / "data/task3" / PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-debug", "calibrate", "build-dataset", "validate-env"):
        child = commands.add_parser(name)
        child.add_argument("--output", type=Path, default=DEFAULT_DATA / "debug" if name in ("build-debug", "calibrate") else DEFAULT_DATA)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifests/val.jsonl")
    evaluation.add_argument("--output", type=Path, required=True)
    plotting = commands.add_parser("plot")
    plotting.add_argument("--run", type=Path, required=True)
    plotting.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.command == "build-debug":
        from .scene_builder import debug_tasks, build_scene
        from .engine import Engine
        from PIL import Image
        args.output.mkdir(parents=True, exist_ok=True)
        records = []
        for name, task in debug_tasks().items():
            path = build_scene(task, args.output / f"{name}.wad")
            task = {**task, "scene_path": str(path.resolve()), "scene_hash": sha256(path), "maze_size": len(task["occupancy"])}
            with Engine() as engine:
                engine.load_task(task, path)
                Image.fromarray(engine.reset()).save(args.output / f"{name}_first_person.png")
                records.append({**task, "spawn_pose": engine.pose()})
        atomic_json(args.output / "tasks.json", records)
        print(json.dumps({"debug_scenes": len(records), "output": str(args.output)}))
    elif args.command == "calibrate":
        from .engine import calibrate
        print(json.dumps(calibrate(args.output), indent=2))
    elif args.command == "build-dataset":
        from .dataset import build_dataset
        from .scene_builder import build_scene
        from .env import MazeEpisodeEnv
        calibration = args.output / "debug/calibration.json"
        if not calibration.exists() or not json.loads(calibration.read_text())["passed"]:
            raise RuntimeError("Run calibrate successfully before dataset creation")
        def progress(row):
            if row["accepted"] % 50 == 0 or row["accepted"] == 1:
                print(json.dumps(row), flush=True)
        factory = functools.partial(MazeEpisodeEnv, calibration=calibration)
        print(json.dumps(build_dataset(args.output, build_scene, factory, progress=progress), indent=2))
    elif args.command == "validate-env":
        from .dataset import load_manifest, reference_controller
        from .env import MazeEpisodeEnv
        results = []
        with MazeEpisodeEnv(calibration=args.output / "debug/calibration.json") as env:
            for split in ("train", "val", "test"):
                tasks = load_manifest(args.output / f"manifests/{split}.jsonl")
                # Controller sanity rechecks only; never evaluates neural policy on test.
                task = tasks[0]
                result = reference_controller(env, task)
                if not result["success"]:
                    raise RuntimeError(f"Reference recheck failed: {split}/{result}")
                results.append({"split": split, "manifest_tasks_checked": len(tasks), "reference_recheck": result})
        atomic_json(args.output / "environment_validation.json", {"passed": True, "splits": results})
        print(json.dumps(results, indent=2))
    elif args.command == "train":
        import yaml
        from .train import train
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        print(json.dumps(train(config, args.output, args.resume), indent=2))
    elif args.command == "evaluate":
        import torch
        from .visual_policy import VisualPolicy
        from .checkpoint import load_checkpoint
        from .dataset import load_manifest
        from .evaluate import evaluate_tasks
        from .env import MazeEpisodeEnv
        if args.output.exists():
            raise FileExistsError(args.output)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        policy = VisualPolicy(hidden_dim=state["config"]["model"]["recurrent_hidden_dim"], device="cpu")
        load_checkpoint(args.checkpoint, policy)
        data_root = args.manifest.resolve().parent.parent
        with MazeEpisodeEnv(calibration=data_root / "debug/calibration.json") as env:
            result = evaluate_tasks(env, policy, load_manifest(args.manifest))
        atomic_json(args.output, result)
    elif args.command == "plot":
        from .plotting import plot_run
        print(json.dumps(plot_run(args.run, args.data_root), indent=2))


if __name__ == "__main__":
    main()
