"""Explicit new-run/resume entrypoint for the isolated approved experiment."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import multiprocessing
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        from .runner import run_experiment
        run_experiment(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.output, args.resume)
    else:
        from .reporting import generate_report
        generate_report(args.run)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
