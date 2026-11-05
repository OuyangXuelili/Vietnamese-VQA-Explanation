"""Stage verified images for a frozen preference experiment."""

import csv
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.utils.environment import load_environment
from vqa_explanation.utils.run_provenance import sha256_file

IMAGE_PATH = re.compile(r"images/COCO_(?:train2014|val2014)_\d{12}\.jpg\Z")
METADATA = {"metadata/" + name for name in (
    "all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "full_image_checksums.csv")}


def package_image_subset(environment, preferences, regression_config, output, *,
                         include_validation=False):
    env = load_environment(Path(environment))
    output = Path(output)
    if output.exists() or output.with_suffix(".sha256").exists():
        raise FileExistsError("Use a new subset archive path")
    pairs = list(read_jsonl(Path(preferences) / "accepted.jsonl"))
    preference_manifest = json.loads((Path(preferences) / "manifest.json").read_text("utf-8"))
    if sha256_file(Path(preferences) / "accepted.jsonl") != preference_manifest["files"][
            "accepted.jsonl"]:
        raise ValueError("Frozen preference records changed")
    canonical = {r["question_id"]: r for r in read_jsonl(env["annotations"])}
    ids = {p["source_question_id"] for p in pairs}
    if any(canonical[q]["split"] != "train" for q in ids):
        raise ValueError("Preference images must belong to train")
    regression = yaml.safe_load(Path(regression_config).read_text("utf-8"))
    ids |= {str(c["question_id"]) for c in regression["cases"]}
    if include_validation:
        ids |= {q for q, row in canonical.items() if row["split"] == "validation"}
    rows = [canonical[q] for q in sorted(ids)]
    with env["image_checksums"].open(encoding="utf-8") as stream:
        checksums = {r["image_name"]: r for r in csv.DictReader(stream)}
    files = {}
    for row in rows:
        relative = row["image_path"]
        if not IMAGE_PATH.fullmatch(relative):
            raise ValueError("Unexpected subset image path")
        path = env["image_root"] / relative
        if sha256_file(path) != checksums[row["image_name"]]["sha256"]:
            raise ValueError("Subset image differs from the canonical checksum")
        files[relative] = path
    for name in ("all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl"):
        files["metadata/" + name] = env["annotations"].parent / name
    files["metadata/full_image_checksums.csv"] = env["image_checksums"]
    manifest = {"schema": "vqa_image_subset_v1", "images": len(files) - len(METADATA),
                "preference_pairs": len(pairs), "canonical_annotation_sha256":
                sha256_file(env["annotations"]), "preferences_sha256":
                sha256_file(Path(preferences) / "accepted.jsonl"),
                "regression_config_sha256": sha256_file(Path(regression_config)),
                "files": {name: sha256_file(p) for name, p in sorted(files.items())},
                "scope": "Preference/regression images with full canonical metadata."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_STORED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("subset_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    digest = sha256_file(output)
    output.with_suffix(".sha256").write_text(digest + "  " + output.name + "\n", "utf-8")
    return {"status": "packaged", "archive": str(output), "sha256": digest,
            "images": manifest["images"], "bytes": output.stat().st_size}


def restore_image_subset(archive_path, expected_sha256, destination):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Provide the pinned subset archive SHA-256")
    if sha256_file(archive_path) != expected_sha256:
        raise ValueError("Subset archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "subset_manifest.json" not in names:
            raise ValueError("Duplicate or missing subset manifest")
        manifest = json.loads(archive.read("subset_manifest.json"))
        files = manifest.get("files", {})
        if (manifest.get("schema") != "vqa_image_subset_v1"
                or not isinstance(files, dict) or not METADATA.issubset(files)
                or set(names) != set(files) | {"subset_manifest.json"}
                or type(manifest.get("images")) is not int or manifest["images"] <= 0
                or manifest.get("images") != len(files) - len(METADATA)
                or manifest.get("canonical_annotation_sha256") != files.get("metadata/all.jsonl")
                or not all(name in METADATA or IMAGE_PATH.fullmatch(name) for name in files)):
            raise ValueError("Invalid subset manifest or member paths")
        for name, expected in files.items():
            info = archive.getinfo(name)
            target = (destination / name).resolve()
            if (stat.S_ISLNK(info.external_attr >> 16)
                    or not target.is_relative_to(destination)):
                raise ValueError("Subset member escapes the extraction directory")
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Subset member checksum mismatch")
            if target.exists() and (not target.is_file() or sha256_file(target) != expected):
                raise ValueError("Existing data differs; choose a new extraction directory")
        for name, expected in files.items():
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
            if sha256_file(target) != expected:
                raise ValueError("Restored subset checksum mismatch")
    return {"status": "verified", "images": manifest["images"],
            "scope": "subset_only", "full_corpus_verified": False,
            "canonical_annotation_sha256": manifest["canonical_annotation_sha256"],
            "preferences_sha256": manifest["preferences_sha256"],
            "extracted_to": str(destination)}
