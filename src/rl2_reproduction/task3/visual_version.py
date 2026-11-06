"""Versioned material-only replacement of frozen WADs, preserving task semantics."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

from PIL import Image
import numpy as np

from .scene_builder import _write_wad
from .visual_assets import read_wad, WadAssets

VERSION = "rl2-task3-visual-stone-v1"
SELECTION = {"RLWALL": ("BRICK9", "texture"), "RLFLOOR": ("FLOOR0_6", "flat"), "RLCEIL": ("FLAT2", "flat")}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_materials(output):
    import vizdoom
    output = Path(output).resolve()
    path = output / "materials.json"
    if path.exists():
        raise FileExistsError(path)
    assets = WadAssets(Path(vizdoom.__file__).parent / "freedoom2.wad")
    config = {"visual_version": VERSION, "vizdoom_version": vizdoom.__version__,
              "freedoom_version": assets.by_name["FREEDOOM"].data.decode().strip(),
              "materials": {}, "texture_scale": "1 source pixel per Doom unit; original default scales unchanged",
              "lightlevel": 240, "goal": "original RLGOAL red floor unchanged; vertical curtains preview only",
              "selection_basis": "Human comparison of exported local candidates with RL2 Figure4; no neural policy or test scores"}
    (output / "materials").mkdir(parents=True, exist_ok=True)
    license_source = Path(__file__).resolve().parents[3] / "licenses/COPYING.Freedoom.v0.13.0.adoc"
    license_target = output / "materials/COPYING.Freedoom.v0.13.0.adoc"
    license_target.write_bytes(license_source.read_bytes())
    config["license"] = {"path": str(license_target.relative_to(output)), "sha256": digest(license_target),
        "source_url": "https://raw.githubusercontent.com/freedoom/freedoom/v0.13.0/COPYING.adoc"}
    for alias, (name, kind) in SELECTION.items():
        image = assets.decode_texture(name) if kind == "texture" else assets.decode_flat(name)
        if image.shape[-1] == 4:
            if not np.all(image[:,:,3] == 255):
                raise ValueError("Selected wall must be fully opaque")
            image = image[:,:,:3]
        destination = output / "materials" / (alias + ".png")
        Image.fromarray(image).save(destination)
        config["materials"][alias] = {**assets.resource_record(name, kind),
            "embedded_alias": alias, "png_path": destination.relative_to(output).as_posix(),
            "png_sha256": digest(destination), "decoded_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def load_material_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["visual_version"] != VERSION:
        raise ValueError("Unsupported material version")
    config["_root"] = str(path.parent)
    return config


@lru_cache(maxsize=16)
def _material_bytes(root, relative, expected):
    path = Path(root) / relative
    if digest(path) != expected:
        raise ValueError(f"Material hash mismatch: {path}")
    return path.read_bytes()


def build_visual_scene(source, destination, config, target_preview=False):
    if target_preview:
        raise ValueError("Vertical target is a separate preview, not part of material version")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Visual variants require a new output path")
    lumps = read_wad(source)
    replacements = {alias: _material_bytes(config["_root"], record["png_path"], record["png_sha256"])
                    for alias, record in config["materials"].items()}
    if set(replacements) != {"RLWALL", "RLFLOOR", "RLCEIL"}:
        raise ValueError("Only wall/floor/ceiling aliases may change")
    if any(sum(l.name == alias for l in lumps) != 1 for alias in replacements):
        raise ValueError("Source material aliases missing/ambiguous")
    _write_wad(destination, [(l.name, replacements.get(l.name, l.data)) for l in lumps])
    return destination


def build_visual_dataset(source, output, config):
    """Repackage existing definitions; never resample or call a neural policy."""
    source, output = Path(source).resolve(), Path(output).resolve()
    manifests = output / "manifests"
    if manifests.exists():
        raise FileExistsError("Refusing an existing visual dataset")
    manifests.mkdir(parents=True)
    summary = {"visual_version": VERSION, "source_root": str(source), "task_protocol": "rl2-task3-from-scratch-v2",
               "materials_sha256": digest(output / "materials.json"), "splits": {},
               "physical_validation": "Inherited original per-task reference-controller results; geometry unchanged; new debug regression recorded separately",
               "source_manifest_sha256": {}, "generator_commit": None}
    for split in ("train", "val", "test"):
        source_path = source / "manifests" / (split + ".jsonl")
        rows = [json.loads(line) for line in source_path.read_text().splitlines()]
        summary["source_manifest_sha256"][split] = digest(source_path)
        converted = []
        for task in rows:
            old_scene = source / task["scene_path"]
            if digest(old_scene) != task["scene_hash"]:
                raise ValueError("Original source scene changed")
            target = output / task["scene_path"]
            build_visual_scene(old_scene, target, config)
            converted.append({**task, "visual_version": VERSION, "source_scene_hash": task["scene_hash"],
                              "scene_hash": digest(target), "materials_sha256": summary["materials_sha256"]})
        path = manifests / (split + ".jsonl")
        path.write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in converted), encoding="utf-8")
        summary["splits"][split] = {"task_count": len(converted), "layout_count": len({t["layout_hash"] for t in rows}),
            "symmetry_group_count": len({t["canonical_group_id"] for t in rows}), "manifest_sha256": digest(path)}
    summary["builder_source_sha256"] = digest(__file__)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
