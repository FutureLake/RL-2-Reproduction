"""Export unsmoothed scientific figures from actual Task 3 run records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _plot_modules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    return plt, ListedColormap


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float(value):
    return np.nan if value is None else float(value)


def _draw_map(ax, task):
    _, ListedColormap = _plot_modules()
    grid = np.asarray(task["occupancy"], dtype=int).copy()
    grid[tuple(task["goal_cell"])] = 2
    size = grid.shape[0]
    ax.imshow(grid, cmap=ListedColormap(["#f3f3ed", "#353b44", "#d62728"]),
              vmin=0, vmax=2, interpolation="nearest", extent=(0, size, size, 0))
    ax.set_xticks(np.arange(size + 1), minor=True)
    ax.set_yticks(np.arange(size + 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.4)
    row, col = task["start_cell"]
    yaw = np.deg2rad(task["start_yaw_deg"])
    ax.scatter([col + 0.5], [row + 0.5], marker="o", s=70, color="#f0b429", edgecolor="black", label="Start")
    ax.arrow(col + 0.5, row + 0.5, 0.38 * np.cos(yaw), -0.38 * np.sin(yaw),
             width=0.035, color="#f0b429", length_includes_head=True)
    ax.set(xlabel="Grid column", ylabel="Grid row", xlim=(0, size), ylim=(size, 0))
    ax.set_aspect("equal")


def plot_scene_preview(task: dict, rgb_image, output: str | Path) -> str:
    """Pair geometry with a caller-supplied *real engine* frame from this task."""
    plt, _ = _plot_modules()
    frame = np.asarray(rgb_image)
    if frame.ndim != 3:
        raise ValueError("preview requires an RGB image")
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = frame.transpose(1, 2, 0)
    if frame.shape[-1] != 3:
        raise ValueError("preview image must have three RGB channels")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    _draw_map(axes[0], task)
    axes[0].set_title("Occupancy / start / red goal floor")
    axes[1].imshow(frame)
    axes[1].set_title("ViZDoom RGB (actual engine frame)")
    axes[1].axis("off")
    fig.suptitle(str(task.get("task_id", "debug scene")))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output.resolve())


def _draw_trace(ax, task, trace, episodes, reset_trace=None):
    _draw_map(ax, task)
    size = task["maze_size"]
    colors = ("#0072b2", "#009e73")
    for index in range(2):
        points = [row["pose"] for row in trace if row["episode_index"] == index and row.get("pose") is not None]
        if episodes and episodes[index].get("start_pose"):
            points.insert(0, episodes[index]["start_pose"])
        if points:
            points = np.asarray(points)
            ax.plot(points[:, 0] / 128, size - points[:, 1] / 128, color=colors[index],
                    linewidth=1.4, alpha=0.85, label=f"Episode {index + 1}")
            ax.scatter(points[-1, 0] / 128, size - points[-1, 1] / 128, color=colors[index], marker="x", s=45)
    if reset_trace:
        points = [row["pose"] for row in reset_trace if row["episode_index"] == 1 and row.get("pose") is not None]
        if points:
            points = np.asarray(points)
            ax.plot(points[:, 0] / 128, size - points[:, 1] / 128, color="#cc79a7",
                    linewidth=1.2, linestyle="--", alpha=0.85, label="Episode 2: reset h")
    ax.legend(fontsize=8, loc="upper right")


def plot_run(run: str | Path, data_root: str | Path) -> dict:
    """Generate six panels (four requested metric classes) and genuine traces.

    B1 is explicitly unrun. Missing successful-length statistics remain NaN in
    figures and empty in CSV; no smoothing, interpolation of data, or synthetic
    learning results are introduced.
    """
    from .dataset import load_manifest
    plt, _ = _plot_modules()
    run, data_root = Path(run).resolve(), Path(data_root).resolve()
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not metrics:
        raise ValueError("plotting needs actual training records")
    initial = _read(run / "validation_initial.json")
    final = _read(run / "validation_final.json")
    summary = _read(run / "summary.json")
    if not summary.get("completed"):
        raise ValueError("S4 plots require a completed run")
    steps = [row["training_steps"] for row in metrics]
    if steps[-1] != summary["training_steps"] or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("inconsistent training-step history")
    validation = [(0, initial), (steps[-1], final)]
    train_summaries = [row["rollout"] for row in metrics]
    output = run / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 13.2), constrained_layout=True)
    axes = axes.ravel()
    val_x = [step for step, _ in validation]
    axes[0].plot(steps, [row["mean_trial_return"] for row in train_summaries], "o-", label="Training batch (8 trials)")
    axes[0].plot(val_x, [result["policy"]["mean_trial_return"] for _, result in validation], "s--", label="Fixed validation subset")
    axes[0].set(title="Trial return", ylabel="Undiscounted return")
    for index, color in enumerate(("#0072b2", "#009e73")):
        label = f"Episode {index + 1}"
        axes[1].plot(steps, [row["episodes"][index]["success_rate"] for row in train_summaries], "o-", color=color, label=f"Train {label}")
        axes[1].plot(val_x, [result["policy"]["episodes"][index]["success_rate"] for _, result in validation], "s--", color=color, label=f"Val {label}")
        axes[2].plot(steps, [_float(row["episodes"][index]["mean_success_length"]) for row in train_summaries], "o-", color=color, label=f"Train {label}")
        axes[2].plot(val_x, [_float(result["policy"]["episodes"][index]["mean_success_length"]) for _, result in validation], "s--", color=color, label=f"Val {label}")
    axes[1].set(title="Episode success rate", ylabel="Fraction", ylim=(-0.03, 1.03))
    axes[2].set(title="Actions in successful episodes only", ylabel="Policy actions")
    success_counts = [row["success_count"] for stats in train_summaries + [initial["policy"], final["policy"]] for row in stats["episodes"]]
    if sum(success_counts) == 0:
        axes[2].text(0.5, 0.5, "No successful episodes (n = 0)\nLength statistic is undefined", transform=axes[2].transAxes,
                     ha="center", va="center", color="#555555")
    axes[3].plot(steps, [_float(row["paired_success_mean_length_improvement"]) for row in train_summaries], "o-", label="Training both-success pairs")
    axes[3].plot(val_x, [_float(result["policy"]["paired_success_mean_length_improvement"]) for _, result in validation], "s--", label="Validation both-success pairs")
    axes[3].axhline(0, color="gray", linewidth=0.8)
    axes[3].set(title="Paired length improvement: L1 - L2", ylabel="Policy actions (both successful)")
    both_counts = [row["both_success_count"] for row in train_summaries] + [initial["policy"]["both_success_count"], final["policy"]["both_success_count"]]
    if sum(both_counts) == 0:
        axes[3].text(0.5, 0.5, "No both-success pairs (n = 0)\nPaired improvement is undefined", transform=axes[3].transAxes,
                     ha="center", va="center", color="#555555")
    for group, style, color, label in (("policy", "o-", "#0072b2", "RL2: retain h"), ("reset", "s--", "#cc79a7", "Same policy: reset h")):
        if all(group in result for _, result in validation):
            axes[4].plot(val_x, [result[group]["episodes"][1]["mean_return"] for _, result in validation], style, color=color, label=label)
            axes[5].plot(val_x, [result[group]["episodes"][1]["success_rate"] for _, result in validation], style, color=color, label=label)
    axes[4].set(title="Episode 2: retained vs reset state", ylabel="Undiscounted return")
    axes[5].set(title="Episode 2: retained vs reset success", ylabel="Fraction", ylim=(-0.03, 1.03))
    for ax in axes[4:]:
        ax.text(0.02, 0.04, "B1 separately trained baseline: not run (S6)", transform=ax.transAxes, fontsize=9, color="#555555")
    for ax in axes:
        ax.set_xlabel("Cumulative training environment steps")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Task 3 bounded smoke: real data, unsmoothed\nSingle training seed; descriptive validation only", fontsize=15)
    curves = output / "learning_curves.png"
    fig.savefig(curves, dpi=160)
    plt.close(fig)

    # One tidy row per actually observed summary, including undefined values.
    columns = ["source", "training_steps", "group", "trial_count", "mean_trial_return", "episode1_success_count",
               "episode2_success_count", "episode1_success_rate", "episode2_success_rate", "episode1_mean_success_length",
               "episode2_mean_success_length", "both_success_count", "paired_success_mean_length_improvement",
               "both_failed", "first_failed_second_succeeded", "first_succeeded_second_failed", "both_succeeded"]
    csv_path = output / "learning_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        rows = [("train", step, "policy", stats) for step, stats in zip(steps, train_summaries)]
        rows += [("validation", step, group, result[group]) for step, result in validation for group in ("policy", "reset") if group in result]
        for source, step, group, stats in rows:
            row = {"source": source, "training_steps": step, "group": group,
                   **{name: stats[name] for name in ("trial_count", "mean_trial_return", "both_success_count", "paired_success_mean_length_improvement")},
                   **stats["outcome_counts"]}
            for index in range(2):
                for name in ("success_count", "success_rate", "mean_success_length"):
                    row[f"episode{index + 1}_{name}"] = stats["episodes"][index][name]
            writer.writerow(row)

    tasks = {task["task_id"]: task for split in ("train", "val") for task in load_manifest(data_root / "manifests" / f"{split}.jsonl")}
    examples = []
    example_path = run / "trial_example.json"
    if example_path.exists():
        example = _read(example_path)
        examples.append(("First training update", example["task_id"], example["trace"], example["episodes"], None))
    for example in final.get("trials", []):
        if example.get("trace"):
            examples.append(("Final validation", example["task_id"], example["trace"], example["episode_summaries"], example.get("reset_trace")))
        if len(examples) == 3:
            break
    trajectory = None
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(5.0 * len(examples), 5.3), constrained_layout=True, squeeze=False)
        for ax, (label, task_id, trace, episodes, reset_trace) in zip(axes[0], examples):
            if task_id not in tasks:
                raise ValueError(f"trace task absent from frozen dataset: {task_id}")
            _draw_trace(ax, tasks[task_id], trace, episodes, reset_trace)
            ax.set_title(f"{label}\n{task_id}\nE1/E2: " + "/".join("success" if item["success"] else "timeout" for item in episodes), fontsize=10)
        fig.suptitle("Actual engine poses; privileged geometry used only for diagnostics", fontsize=12)
        trajectory = output / "trajectory_examples.png"
        fig.savefig(trajectory, dpi=160)
        plt.close(fig)
    result = {"learning_curves": str(curves), "learning_csv": str(csv_path),
              "trajectory_examples": str(trajectory) if trajectory else None,
              "smoothing_window": 1, "b1": "not run", "source": "actual metrics and evaluation records"}
    (output / "plot_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
