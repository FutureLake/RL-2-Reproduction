"""Generate navigation data locally, then apply the stone/goal visual recipe."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

from rl2_reproduction.task3.utils import PROJECT
from rl2_reproduction.task3.visual_version import freeze_materials, load_material_config, build_visual_dataset
from rl2_reproduction.task3.internal_goal import VERSION, SPEC, write_internal_goal
from rl2_reproduction.task3.scene_builder import debug_tasks


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def build_visual():
    raw = PROJECT / "data/task3/rl2-task3-from-scratch-v2"
    stone = PROJECT / "data/task3/rl2-task3-visual-stone-v1"
    final = PROJECT / "data/task3" / VERSION
    if stone.exists() or final.exists():
        raise FileExistsError("Visual output already exists; use a fresh working copy")
    material = freeze_materials(stone)
    build_visual_dataset(raw, stone, load_material_config(material))
    final.mkdir()
    shutil.copytree(stone / "materials", final / "materials")
    shutil.copyfile(material, final / "base_materials.json")
    dump(final / "visual_config.json", {
        "visual_version": VERSION, "base_visual_version": "rl2-task3-visual-stone-v1",
        "spec": SPEC, "base_material_config_sha256": sha(material),
        "recipe": "public deterministic stone materials and nonblocking inset goal faces",
    })
    hashes, splits = {}, {}
    for split in ("train", "val", "test"):
        rows = [json.loads(x) for x in (stone / f"manifests/{split}.jsonl").read_text().splitlines()]
        for row in rows:
            source, dest = stone / row["scene_path"], final / row["scene_path"]
            if sha(source) != row["scene_hash"]:
                raise ValueError("Source scene hash mismatch")
            write_internal_goal(source, dest, row)
            row.update(visual_version=VERSION, source_scene_hash=row["scene_hash"], scene_hash=sha(dest),
                       visual_config_sha256=sha(final / "visual_config.json"))
        path = final / f"manifests/{split}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        hashes[split] = sha(path)
        splits[split] = {"task_count": len(rows), "symmetry_group_count": len({r["canonical_group_id"] for r in rows})}
    (final / "debug").mkdir()
    shutil.copyfile(raw / "debug/calibration.json", final / "debug/calibration.json")
    for name, task in debug_tasks().items():
        # Debug views use the same material transform as all train/val/test scenes.
        from rl2_reproduction.task3.visual_version import build_visual_scene
        dest = stone / "debug" / f"{name}.wad"
        build_visual_scene(raw / "debug" / f"{name}.wad", dest, load_material_config(material))
        write_internal_goal(dest, final / "debug" / f"{name}.wad", task)
    dump(final / "dataset_summary.json", {"visual_version": VERSION, "manifest_sha256": hashes,
         "splits": splits, "controller_validation": "inherited from raw scenes; visual transform preserves collision geometry"})
    print(json.dumps({"data_root": str(final), "manifest_sha256": hashes}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-only", action="store_true", help="Use an already generated/calibrated raw dataset")
    args = parser.parse_args()
    if not args.visual_only:
        for stage in ("build-debug", "calibrate", "build-dataset"):
            subprocess.run([sys.executable, "-m", "rl2_reproduction.task3.cli", stage], check=True)
    build_visual()
