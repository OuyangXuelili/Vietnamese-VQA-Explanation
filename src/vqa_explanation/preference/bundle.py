"""Verify candidate review exports before rendering or converting labels."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from vqa_explanation.inference.export import (
    read_jsonl,
    select_records,
    validate_saved_predictions,
)
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.utils.run_provenance import sha256_file


def verify_bundle(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Bundle SHA-256 differs from the Colab output")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "checksums.json" not in names:
            raise ValueError("Missing checksums or duplicate ZIP members")
        for name in names:
            parts = PurePosixPath(name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or "\\" in name
                or ":" in name
                or name != parts.as_posix()
            ):
                raise ValueError(f"Unsafe ZIP member: {name}")
        data = {name: archive.read(name) for name in names}
    checks = json.loads(data["checksums.json"])
    if set(data) != set(checks) | {"checksums.json"}:
        raise ValueError("Bundle members differ from the checksum manifest")
    if any(hashlib.sha256(data[name]).hexdigest() != digest for name, digest in checks.items()):
        raise ValueError("Member checksum changed")
    inputs = [json.loads(line) for line in data["inputs.jsonl"].decode().splitlines()]
    if not inputs:
        raise ValueError("Empty candidate bundle")
    predictions = [json.loads(line) for line in data["candidates.jsonl"].decode().splitlines()]
    manifest = json.loads(data["candidates.jsonl.manifest.json"])
    if (
        manifest["inputs"]["split"] != "train"
        or manifest["inputs"]["context"]["purpose"] != "preference_candidates_pending_review"
        or any(row["split"] != "train" for row in inputs)
    ):
        raise ValueError("Only unreviewed train candidates belong in this review page")
    completed = validate_saved_predictions(predictions, inputs, manifest)
    if len(completed) != len(inputs):
        raise ValueError("Incomplete candidate bundle")
    for row in inputs:
        digest = hashlib.sha256(data[row["image_path"]]).hexdigest()
        if digest != manifest["inputs"]["image_hashes"][row["question_id"]]:
            raise ValueError("Image differs from the prediction manifest")
    return data, inputs, {row["question_id"]: row for row in predictions}


def _json_object(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _safe_member(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"Unsafe image path: {name!r}")
    parts = PurePosixPath(name)
    if (parts.is_absolute() or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe image path: {name}")
    return parts


def _add_image(sources, name, root):
    parts = _safe_member(name)
    reserved = {"checksums.json", "index.html", "review_notes.jsonl", *sources}
    lowered = name.casefold()
    for existing in reserved:
        other = existing.casefold()
        if (lowered == other or lowered.startswith(other + "/")
                or other.startswith(lowered + "/")):
            raise ValueError(f"Image path collides with a bundle member: {name}")
    source = (root / parts).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(f"Image path escapes image root or is missing: {name}")
    sources[name] = source


def _review_sources(
    candidate_dir, pool_dir, m1_selection, image_root, *, expected_count,
    annotations_sha256, pool_questions_sha256, pool_manifest_sha256, m1_selection_sha256,
):
    if type(expected_count) is not int or not 1 <= expected_count <= 2000:
        raise ValueError("expected_count must be an integer between 1 and 2000")
    anchors = (annotations_sha256, pool_questions_sha256, pool_manifest_sha256,
               m1_selection_sha256)
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in anchors):
        raise ValueError("Expected SHA-256 anchors must be lowercase hexadecimal digests")
    candidate_dir, pool_dir = Path(candidate_dir), Path(pool_dir)
    sources = {name: candidate_dir / name for name in (
        "inputs.jsonl", "candidates.jsonl", "candidates.jsonl.manifest.json", "summary.json",
    )}
    sources.update({
        "pool/questions.jsonl": pool_dir / "questions.jsonl",
        "pool/manifest.json": pool_dir / "manifest.json",
        "m1_selection.json": Path(m1_selection),
    })
    checks = {name: sha256_file(path) for name, path in sources.items()}
    for name, digest in (
        ("pool/questions.jsonl", pool_questions_sha256),
        ("pool/manifest.json", pool_manifest_sha256),
        ("m1_selection.json", m1_selection_sha256),
    ):
        if checks[name] != digest:
            raise ValueError(f"Frozen anchor hash changed: {name}")
    summary = _json_object(sources["summary.json"])
    if (summary.get("status") != "generated_pending_review"
            or type(summary.get("n")) is not int or summary["n"] != expected_count):
        raise ValueError("Summary must contain the expected complete pending-review count")
    for field, name in (
        ("inputs_sha256", "inputs.jsonl"),
        ("predictions_sha256", "candidates.jsonl"),
        ("manifest_sha256", "candidates.jsonl.manifest.json"),
    ):
        if summary.get(field) != checks[name]:
            raise ValueError(f"Summary hash changed: {name}")
    pool = _json_object(sources["pool/manifest.json"])
    questions = read_jsonl(sources["pool/questions.jsonl"])
    if (pool.get("schema_version") != 1
            or pool.get("status") != "questions_selected_not_reviewed"
            or pool.get("annotations_sha256") != annotations_sha256
            or type(pool.get("count")) is not int or pool["count"] != len(questions)
            or expected_count > len(questions)
            or pool.get("files", {}).get("questions.jsonl") != pool_questions_sha256
            or pool.get("calibration_question_ids") != [
                row.get("source_question_id") for row in questions[:100]
            ]):
        raise ValueError("Frozen pool manifest differs from its references/count/hash")
    for field in ("source_question_id", "image_id", "image_path"):
        values = [row.get(field) for row in questions]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"Frozen pool has invalid {field}")
        if len(set(values)) != len(values):
            raise ValueError(f"Frozen pool has duplicate {field}")
    for row in questions:
        _safe_member(row["image_path"])
        if (row.get("split") != "train" or row.get("review_status") != "pending"
                or row.get("annotations_sha256") != annotations_sha256
                or row.get("prompt") != build_user_prompt(row.get("question", ""))):
            raise ValueError("Frozen pool must contain pending canonical train questions")
    records = read_jsonl(sources["inputs.jsonl"])
    if len(records) != expected_count:
        raise ValueError("Input count differs from expected_count")
    for record, question in zip(records, questions[:expected_count], strict=True):
        expected = {
            "question_id": question["source_question_id"], "split": "train",
            "image_id": question["image_id"], "image_path": question["image_path"],
            "question": question["question"], "answer": question["reference_answer"],
            "explanations": question["reference_explanations"],
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise ValueError("Inputs differ from the exact frozen train prefix")
    manifest = _json_object(sources["candidates.jsonl.manifest.json"])
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, dict) or not isinstance(inputs.get("context"), dict):
        raise ValueError("Candidate manifest is missing the frozen review context")
    context = inputs.get("context", {})
    expected_context = {
        "purpose": "preference_candidates_pending_review", "count": expected_count,
        "selection_order": "frozen_prefix", "annotations_sha256": annotations_sha256,
        "selection_sha256": pool_questions_sha256,
        "selection_manifest_sha256": pool_manifest_sha256,
        "m1_selection_sha256": m1_selection_sha256,
    }
    if type(inputs.get("seed")) is not int:
        raise ValueError("Candidate manifest requires its original integer seed")
    selected_digest = hashlib.sha256(json.dumps(
        select_records(records, "train", 0, inputs.get("seed")),
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    if (manifest.get("schema_version") != 1 or inputs.get("split") != "train"
            or inputs.get("experiment_id") != "M1_SFT" or inputs.get("limit") != 0
            or inputs.get("annotations_sha256") != checks["inputs.jsonl"]
            or inputs.get("selected_sha256") != selected_digest
            or any(context.get(field) != value for field, value in expected_context.items())):
        raise ValueError("Candidate manifest differs from the frozen review context")
    files = inputs.get("adapter_files")
    if not isinstance(files, dict) or set(files) != {
        "adapter_config.json", "adapter_model.safetensors",
    }:
        raise ValueError("Candidate manifest is missing the selected M1 adapter hashes")
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in files.values()):
        raise ValueError("Candidate manifest has invalid M1 adapter hashes")
    parent = {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()}
    selection = _json_object(sources["m1_selection.json"])
    if (selection.get("status") != "selected"
            or selection.get("selection_split") != "validation"
            or selection.get("annotations_sha256") != annotations_sha256
            or selection.get("model_id") != inputs.get("model_id")
            or selection.get("model_revision") != inputs.get("model_revision")
            or selection.get("adapter") != parent or context.get("m1_adapter") != parent):
        raise ValueError("Candidate adapter differs from the frozen M1 selection")
    predictions = read_jsonl(sources["candidates.jsonl"])
    completed = validate_saved_predictions(predictions, records, manifest)
    if len(completed) != expected_count:
        raise ValueError("Incomplete candidate bundle")
    if set(inputs.get("image_hashes", {})) != completed:
        raise ValueError("Candidate manifest image IDs differ from the frozen train prefix")
    root = Path(image_root).resolve()
    for row in questions[:expected_count]:
        name, qid = row["image_path"], row["source_question_id"]
        _add_image(sources, name, root)
        digest = sha256_file(sources[name])
        if digest != row.get("image_sha256") or digest != inputs["image_hashes"][qid]:
            raise ValueError(f"Selected image hash changed: {qid}")
        checks[name] = digest
    return sources, checks


def _same_bytes(left, right):
    while True:
        chunk = left.read(1024 * 1024)
        if chunk != right.read(1024 * 1024):
            return False
        if not chunk:
            return True


def package_preference_review(
    candidate_dir, pool_dir, m1_selection, image_root, output, *, expected_count,
    annotations_sha256, pool_questions_sha256, pool_manifest_sha256, m1_selection_sha256,
):
    """Package completed candidates and anchored evidence, without assigning labels."""
    sources, checks = _review_sources(
        candidate_dir, pool_dir, m1_selection, image_root, expected_count=expected_count,
        annotations_sha256=annotations_sha256, pool_questions_sha256=pool_questions_sha256,
        pool_manifest_sha256=pool_manifest_sha256, m1_selection_sha256=m1_selection_sha256,
    )
    checksum_payload = (json.dumps(checks, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n").encode("utf-8")
    output = Path(output)
    if output.exists():
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            if (len(names) != len(sources) + 1
                    or set(names) != set(sources) | {"checksums.json"}
                    or archive.read("checksums.json") != checksum_payload):
                raise FileExistsError(f"Existing review bundle differs: {output}")
            for name, path in sources.items():
                with archive.open(name) as saved, path.open("rb") as source:
                    if not _same_bytes(saved, source):
                        raise FileExistsError(f"Existing review bundle differs: {output}: {name}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(sources.items()):
                archive.write(path, name)
            archive.writestr("checksums.json", checksum_payload)
    digest = sha256_file(output)
    verify_bundle(output, digest)
    return {"path": str(output.resolve()), "sha256": digest, "count": expected_count,
            "status": "generated_pending_review"}
