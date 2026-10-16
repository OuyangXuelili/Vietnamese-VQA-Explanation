"""Translate a declared cloud launch into the existing preference-stage CLI."""

import argparse
import hashlib
import json
import re
import stat
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from vqa_explanation.training.preference_data import prepare_stage
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def restore_preference_bundle(archive_path, expected_sha256, destination):
    """Restore a verified metadata bundle without replacing an earlier dataset."""
    archive_path, destination = Path(archive_path), Path(destination)
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256_file(archive_path) != expected_sha256):
        raise ValueError("Preference archive checksum mismatch")
    if destination.is_symlink():
        raise ValueError("Preference destination cannot be linked")
    destination = destination.resolve()
    required = {f"frozen/{name}" for name in (
        "accepted.jsonl", "chosen_only.jsonl", "manifest.json", "quality_review.jsonl",
        "decision_record.json", "readiness.json", "text_only.json",
    )} | {"training_lock.json"}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ValueError("Preference bundle is incomplete")
        for name, digest in manifest.items():
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or not (name in required or name.startswith("evidence/"))
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("Unsafe preference bundle entry")
        expected_names = set(manifest) | {"BUNDLE_MANIFEST.json"}
        if (len(archive.namelist()) != len(expected_names)
                or set(archive.namelist()) != expected_names or archive.testzip() is not None):
            raise ValueError("Preference bundle members/CRC differ")
        if sum(member.file_size for member in archive.infolist()) > 256 * 1024 * 1024:
            raise ValueError("Preference metadata bundle is too large")
        for member in archive.infolist():
            if (member.is_dir() or member.file_size > 64 * 1024 * 1024
                    or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK):
                raise ValueError("Linked, directory or oversized preference member")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Preference bundle member changed: {name}")
        if destination.exists():
            for name, digest in manifest.items():
                path = destination / name
                parts = Path(name).parts
                if (any(destination.joinpath(*parts[:i]).is_symlink()
                        for i in range(1, len(parts) + 1))
                        or not path.is_file() or sha256_file(path) != digest):
                    raise ValueError(f"Existing preference data differs: {name}")
            if (destination / "BUNDLE_MANIFEST.json").is_symlink() or json.loads(
                (destination / "BUNDLE_MANIFEST.json").read_text("utf-8")
            ) != manifest:
                raise ValueError("Existing preference manifest differs")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="preferences-", dir=destination.parent) as temporary:
                archive.extractall(temporary)
                Path(temporary).rename(destination)
    return destination


def validate_dataset_lock(record, project_dir, preferences):
    """Bind the supplied bundle to the pinned curation decision."""
    path_value = record.get("preference_decision_record")
    digest = record.get("preference_decision_record_sha256")
    if path_value is None and digest is None:
        return None
    if (not isinstance(path_value, str) or not path_value.strip()
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Dataset lock requires a decision path and SHA-256")
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(project_dir) / path
    if sha256_file(path) != digest:
        raise ValueError("Pinned preference decision changed")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if (decision.get("schema") != "preference_dataset_decision_v1"
            or decision.get("data_locked") is not True
            or type(decision.get("pairs")) is not int
            or decision["pairs"] != record["expected_pairs"]):
        raise ValueError("Preference decision differs from the locked launch")
    if sha256_file(Path(preferences) / "manifest.json") != decision.get("frozen_manifest_sha256"):
        raise ValueError("Preference bundle differs from the pinned decision")
    return digest


def load_launch(path, project_dir):
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("schema") != "preference_training_launch_v1"
            or record.get("scope") != "main_preference_experiment"):
        raise ValueError("Use a main preference launch record, not an abandoned pilot")
    if record.get("data_locked") is not True:
        return record, None
    if type(record.get("expected_pairs")) is not int or record["expected_pairs"] <= 0:
        raise ValueError("Locked launch requires an explicit positive pair count")
    paths = {}
    for key in ("preferences", "m1_adapter", "m1_selection"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Launch needs {key}")
        path_value = Path(value)
        paths[key] = path_value if path_value.is_absolute() else project_dir / path_value
    for key in ("environment", "study_config"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Launch needs {key}")
        path_value = Path(value)
        paths[key] = path_value if path_value.is_absolute() else project_root() / path_value
    return record, paths


def prepare_launch(launch_path, project_dir, arm, mode,
                   output_dir=None, gpu_gate=None, resume=None):
    if arm not in {"m2", "m3"} or mode not in {"inspect", "gate", "train"}:
        raise ValueError("Unknown preference arm or execution mode")
    if mode == "inspect" and resume is not None:
        raise ValueError("Resume requires a training mode")
    record, paths = load_launch(launch_path, project_dir)
    if paths is None:
        return {"status": "blocked", "reason": "Preference dataset and final N are not locked",
                "main_training_started": False}, None
    decision_hash = validate_dataset_lock(record, project_dir, paths["preferences"])
    config_key = "m2_config" if arm == "m2" else "m3_config"
    relative = record.get(config_key)
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"Launch needs {config_key}")
    args = argparse.Namespace(
        **paths, config=project_root() / relative,
        model_config=project_root() / "configs/model/qwen25_vl_3b.yaml",
        output_dir=output_dir, preflight_only=mode == "inspect",
        pilot_steps=5 if mode == "gate" else 0,
        gpu_gate=gpu_gate, resume_from_checkpoint=resume, stop_after_step=None,
    )
    plan = prepare_stage(args)[2]
    if plan["pairs"] != record["expected_pairs"]:
        raise ValueError("Launch N differs from both config and frozen dataset")
    if mode != "inspect" and output_dir is None:
        raise ValueError("Training requires a new explicit output directory")
    if mode == "train" and gpu_gate is None:
        raise ValueError("Main training requires the matching GPU gate")
    return {"status": "metadata_passed", "plan": plan, "gpu_training_verified": False,
            "preference_decision_sha256": decision_hash}, args


def stage_output(run_folder, arm, mode, attempt="initial"):
    """Separate resumed output from the original run and its GPU gate."""
    if arm not in {"m2", "m3"} or mode not in {"inspect", "gate", "train"}:
        raise ValueError("Unknown preference arm or execution mode")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", attempt):
        raise ValueError("Use a short attempt label without path separators")
    if attempt != "initial" and mode != "train":
        raise ValueError("Attempt labels apply only to main training")
    suffix = "" if attempt == "initial" else f"_{attempt}"
    return run_folder / f"{arm}_{mode}{suffix}"


def launch_command(repo, record, project_dir, arm, mode, run_folder,
                   resume=None, attempt="initial", python_executable=None):
    """Keep notebook cells limited to one explicit arm invocation."""
    output = stage_output(run_folder, arm, mode, attempt)
    if resume is not None and (mode != "train" or attempt == "initial"):
        raise ValueError("Resume requires train mode and a new attempt label")
    command = [str(python_executable or sys.executable),
               str(repo / "scripts/launch_preference_training.py"),
               "--launch-record", str(record), "--project-dir", str(project_dir),
               "--arm", arm, "--mode", mode,
               "--output-dir", str(output)]
    if mode == "train":
        command += ["--gpu-gate", str(run_folder / f"{arm}_gate/training_result.json")]
    if resume is not None:
        command += ["--resume-from-checkpoint", str(resume)]
    return command
