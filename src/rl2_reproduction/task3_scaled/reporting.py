"""Read-only reporting of an aligned run's actual records; no policy execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _num(value):
    return np.nan if value is None else value


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix(".svg"))
    _plt().close(fig)


def _shared_legend(fig, axes, empty=None):
    entries = {}
    for ax in np.asarray(axes, dtype=object).ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.title.set_fontsize(14)
    if entries:
        if empty is not None:
            empty.axis("off")
            empty.legend(list(entries.values()), list(entries), loc="center", frameon=False, fontsize=13)
        else:
            fig.legend(list(entries.values()), list(entries), loc="outside lower center", ncols=min(4, len(entries)), frameon=False, fontsize=12)


def _curve_row(label, steps, group, stats):
    row = {"source": label, "training_steps": steps, "group": group,
           "trial_count": stats.get("trial_count"), "mean_trial_return": stats.get("mean_trial_return"),
           "both_success_count": stats.get("both_success_count"),
           "paired_length_improvement": stats.get("paired_success_mean_length_improvement")}
    for i, episode in enumerate(stats.get("episodes", []), 1):
        for key in ("success_rate", "success_count", "timeout_count", "timeout_rate", "mean_length", "mean_success_length", "collision_fraction", "forward_fraction", "turn_fraction"):
            row[f"episode{i}_{key}"] = episode.get(key)
    return row


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _examples(run, label, result, tasks):
    """Fixed geometry examples plus first task ID in each observed outcome class."""
    if result is None:
        return []
    candidates = sorted((r for r in result.get("trials", []) if r.get("trace") and r["task_id"] in tasks), key=lambda r: r["task_id"])
    if not candidates:
        return []
    selected, used = [], set()
    for group in sorted({tasks[r["task_id"]]["canonical_group_id"] for r in candidates}):
        row = next(r for r in candidates if tasks[r["task_id"]]["canonical_group_id"] == group)
        selected.append(("fixed_geometry", row)); used.add(row["task_id"])
        if len(selected) == 3:
            break
    outcomes = ((False, False), (False, True), (True, False), (True, True))
    for outcome in outcomes:
        rows = [r for r in candidates if tuple(bool(e["success"]) for e in r["episode_summaries"]) == outcome]
        if rows and rows[0]["task_id"] not in used:
            selected.append(("E1_" + str(int(outcome[0])) + "_E2_" + str(int(outcome[1])), rows[0]))
            used.add(rows[0]["task_id"])
    plt = _plt()
    from rl2_reproduction.task3.plotting import _draw_trace
    columns = 3
    rows_count = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 5.8 * rows_count), layout="constrained", squeeze=False)
    fig.set_constrained_layout_pads(h_pad=.18, w_pad=.08, hspace=.12, wspace=.04)
    flat = axes.ravel()
    manifest = []
    for i, (reason, row) in enumerate(selected):
        _draw_trace(flat[i], tasks[row["task_id"]], row["trace"], row["episode_summaries"], row.get("reset_trace"))
        outcomes_text = " / ".join(f"E{j+1}: {e['length']} ({'success' if e['success'] else 'timeout'})" for j,e in enumerate(row["episode_summaries"]))
        flat[i].set_title(f"{reason}\n{row['task_id']}\n{outcomes_text}", fontsize=14)
        destination = run / "examples" / label / row["task_id"]
        _dump(destination / "trajectory.json", row)
        item = {"task_id": row["task_id"], "selection": reason, "trajectory": str((destination / "trajectory.json").relative_to(run)), "artifact": row.get("artifact")}
        manifest.append(item)
        artifact = row.get("artifact", {})
        artifact_path = Path(artifact.get("path", ""))
        if artifact_path.is_file() and artifact_path.suffix == ".npz":
            with np.load(artifact_path, allow_pickle=False) as arrays:
                images = arrays["retain_images_uint8"]
                indices = [0, len(images)//2, len(images)-1]
                picture, axs = plt.subplots(1, 3, figsize=(9, 3.4), layout="constrained")
                for ax, t in zip(axs, indices):
                    ax.imshow(images[t].transpose(1,2,0), interpolation="nearest")
                    ax.set_title(f"Decision {t+1}", fontsize=14); ax.axis("off")
                picture.suptitle(row["task_id"] + " | actual 40 x 30 inputs", fontsize=18)
                _save(picture, destination / "policy_inputs.png")
                item["policy_inputs"] = str((destination / "policy_inputs.png").relative_to(run))
    for ax in flat[len(selected):]:
        ax.axis("off")
    _shared_legend(fig, flat, empty=flat[len(selected)] if len(selected)<len(flat) else None)
    context = "preflight random initialization" if label == "preflight" else "final checkpoint"
    fig.suptitle(f"{label}: actual trajectories | {context}", fontsize=18)
    _save(fig, run / "plots" / f"{label}_trajectories.png")
    _dump(run / "examples" / label / "manifest.json", {"selection_rule":"first sorted task per geometry group (up to three); first sorted task in each E1/E2 success class; duplicates omitted", "examples":manifest})
    return manifest


def generate_report(run_dir, project_root=None):
    run = Path(run_dir).resolve()
    config = _read(run / "config.json")
    if config is None:
        raise FileNotFoundError(run / "config.json")
    project = Path(project_root or config.get("project_root", Path(__file__).resolve().parents[3])).resolve()
    metrics_path = run / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.exists() else []
    summary = _read(run / "summary.json", {})
    status = _read(run / "status.json", {})
    steps = metrics[-1]["training_steps"] if metrics else 0
    data = (project / config["data_root"]).resolve()
    tasks = {}
    for split in ("val", "test"):
        path = data / "manifests" / f"{split}.jsonl"
        if path.exists():
            tasks.update({t["task_id"]:t for t in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())})
    selection = _read(run / "evaluation_tasks.json", {})
    fixed_ids = set(selection.get("intermediate_val", selection.get("fixed_val30", [])))
    if not fixed_ids:
        val_tasks = [t for t in tasks.values() if t["split"] == "val"]
        for group in sorted({t["canonical_group_id"] for t in val_tasks}):
            fixed_ids.update(t["task_id"] for t in sorted([t for t in val_tasks if t["canonical_group_id"] == group], key=lambda t:t["task_id"])[:10])
    paths = [run / "validation_initial.json", *sorted(run.glob("validation_update_*.json")), run / "validation_final.json", run / "test_final.json"]
    # Include optional fixed 500k milestone without silently mixing it into a different subset.
    paths += sorted(p for p in run.glob("validation_milestone_*.json") if p not in paths)
    evaluations = []
    for path in paths:
        if not path.exists():
            continue
        result = _read(path)
        update = result.get("update", 0 if "initial" in path.stem else len(metrics))
        if "update_" in path.stem:
            update = int(path.stem.split("_")[-1])
        evaluated_steps = result.get("training_steps", metrics[update-1]["training_steps"] if update and update <= len(metrics) else 0)
        evaluations.append((path, result, evaluated_steps))
    curves = [_curve_row("train", row["training_steps"], "train", row["rollout"]) for row in metrics if "rollout" in row]
    fixed_curve = []
    from rl2_reproduction.task3.metrics import summarize_trials
    for path, result, evaluated_steps in evaluations:
        for branch in ("policy", "reset"):
            if branch in result:
                curves.append(_curve_row(path.stem, evaluated_steps, branch, result[branch]))
        if path.stem.startswith("validation") and fixed_ids:
            selected = [r for r in result.get("trials", []) if r["task_id"] in fixed_ids]
            if {r["task_id"] for r in selected} == fixed_ids:
                fixed_curve.append(_curve_row(path.stem, evaluated_steps, "fixed_val30", summarize_trials(selected)))
    fixed_curve.sort(key=lambda r:r["training_steps"])
    plots = run / "plots"; plots.mkdir(exist_ok=True)
    _write_csv(plots / "learning_curves.csv", curves + fixed_curve)
    plt = _plt()
    if metrics or fixed_curve:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        training = [r for r in curves if r["group"] == "train"]
        for records, label, style in ((training,"Training batch",".-"),(fixed_curve,"Fixed val30","o--")):
            if not records:
                continue
            x = [r["training_steps"] for r in records]
            axes[0,0].plot(x,[_num(r["mean_trial_return"]) for r in records],style,label=label)
            for ep in (1,2):
                axes[0,1].plot(x,[_num(r.get(f"episode{ep}_success_rate")) for r in records],style,label=f"{label} E{ep}")
                axes[1,0].plot(x,[_num(r.get(f"episode{ep}_mean_success_length")) for r in records],style,label=f"{label} E{ep}")
            axes[1,1].plot(x,[_num(r["paired_length_improvement"]) for r in records],style,label=label)
        for ax,title in zip(axes.ravel(),("Trial return","Episode success","Successful length (conditional)","Both-success length: E1 minus E2")):
            ax.set_title(title,fontsize=14); ax.set_xlabel("Accepted training actions"); ax.grid(alpha=.2)
        axes[0,1].set_ylim(-.03,1.03)
        axes[1,1].axhline(0,color="gray",linewidth=.8)
        fig.suptitle("WeightNorm RL2: measured learning curves",fontsize=18)
        _shared_legend(fig,axes)
        _save(fig,plots / "learning_curves.png")
    diagnostic_keys=("kl","entropy","clip_fraction","value_loss","cnn_parameter_l2_change","gru_parameter_l2_change")
    diagnostics=[{"update":r["update"],"training_steps":r["training_steps"],**{k:r.get(k) for k in diagnostic_keys}} for r in metrics]
    _write_csv(plots / "ppo_diagnostics.csv", diagnostics)
    if diagnostics:
        fig, axes=plt.subplots(3,2,figsize=(12,11),layout="constrained")
        for ax,key in zip(axes.ravel(),diagnostic_keys):
            ax.plot([r["training_steps"] for r in diagnostics],[_num(r[key]) for r in diagnostics],".-")
            ax.set_title(key.replace("_"," "),fontsize=14);ax.set_xlabel("Accepted training actions");ax.grid(alpha=.2)
        fig.suptitle("PPO / shared-network diagnostics",fontsize=18);_save(fig,plots / "ppo_diagnostics.png")
    final_val = next((r for p,r,_ in evaluations if p.stem == "validation_final"),None)
    final_test = next((r for p,r,_ in evaluations if p.stem == "test_final"),None)
    examples={"validation":_examples(run,"validation",final_val,tasks),"test":_examples(run,"test",final_test,tasks)}
    lines=["# Task 3 WeightNorm / PPO：实际运行报告","",
           f"状态：{'completed' if summary.get('completed') else status.get('state',status.get('phase',status.get('status','incomplete')))}。已提交训练动作 {steps:,}；策略更新 {len(metrics)}。",
           f"seed={config.get('run',{}).get('seed','N/A')}；视觉版本 `{config.get('visual_version','N/A')}`；数据目录 `{data}`。",
           "", "0–500万动作沿用父实验，随后继续同一seed并调整学习率；保留PPO。该报告只读取已有结果，没有运行策略、重采环境或调整参数。",
           "", "## 验证与最终测试", "", "| 集合/分支 | n | Trial回报 | E1成功 | E2成功 | E1成功长度(n) | E2成功长度(n) | 双成功长度改善(n) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    paired_notes = []
    for label,result in (("val final",final_val),("test final",final_test)):
        if result is None:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        for key in ("policy","reset"):
            if key not in result:
                continue
            s=result[key];e=s["episodes"]
            lines.append(f"| {label}/{key} | {s['trial_count']} | {_fmt(s['mean_trial_return'])} | {_pct(e[0]['success_rate'])} | {_pct(e[1]['success_rate'])} | {_fmt(e[0]['mean_success_length'],1)} ({e[0]['success_count']}) | {_fmt(e[1]['mean_success_length'],1)} ({e[1]['success_count']}) | {_fmt(s.get('paired_success_mean_length_improvement'),1)} ({s.get('both_success_count',0)}) |")
        paired=result.get("paired_episode2",{})
        if paired:
            paired_notes.append(f"{label} retain−reset：E2回报差 {_fmt(paired.get('mean_return_difference'))}；成功率差 {_fmt(None if paired.get('mean_success_difference') is None else 100*paired['mean_success_difference'],1)} 个百分点。")
    for note in paired_notes:
        lines.extend(["",note])
    lines.extend(["", "reset-h是同一已训练模型的推理消融，非独立训练B1。单seed、相关几何配置仅支持描述性结论；成功长度只统计成功样本，空样本保留N/A。本阶段仅在最终checkpoint评分测试集；但同一测试集的500万结果此前已被查看，不能当作全新盲测。",
                  "", "## 图与原始记录", "", "![学习曲线](plots/learning_curves.png)","", "![PPO诊断](plots/ppo_diagnostics.png)", "", "曲线使用未平滑的真实记录；固定val30的首尾点从相同任务的全量验证中提取。训练任务和验证任务不混为一个总体。"])
    for name, rows in examples.items():
        if rows:
            lines.extend(["",f"![{name}轨迹](plots/{name}_trajectories.png)","",f"选例规则与原始轨迹见 [manifest](examples/{name}/manifest.json)。实际40×30输入仅在历史评估有保存时展示，未重渲染。"])
    lines.extend(["", "## 可追溯性", "", "实际配置、源码/数据哈希、依赖、checkpoint与物理采样账目保存在本运行目录。训练计数、评估动作、工程诊断动作分开记录。", "", "[config.json](config.json) · [metrics.jsonl](metrics.jsonl) · [summary.json](summary.json) · [checkpoint](latest.pt)"])
    (run / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    inputs=[run/"config.json",metrics_path,*[p for p,_,_ in evaluations]]
    manifest={"report":"report.md","report_sha256":_sha(run/"report.md"),"script_sha256":_sha(Path(__file__)),
              "inputs":{p.name:_sha(p) for p in inputs if p.exists()},"no_policy_replay":True,"no_new_environment_actions":True,"examples":examples}
    _dump(run/"report_artifacts.json",manifest)
    return manifest
