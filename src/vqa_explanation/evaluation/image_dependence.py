"""Validation-only image permutation diagnostic."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from filelock import FileLock

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation.answer_metrics import exact_match, normalize_answer
from vqa_explanation.inference.export import (
    export_predictions,
    validate_saved_predictions,
    validate_selected_records,
)
from vqa_explanation.utils.run_provenance import sha256_file


def prepare_images(annotations, image_root, output_dir, count=200, seed=42):
    annotations, image_root, output = Path(annotations), Path(image_root), Path(output_dir)
    if output.exists():
        raise FileExistsError("Preserve the previous image-permutation bundle")
    if type(count) is not int or count < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Require count >= 2 and a non-negative seed")
    annotations_hash = sha256_file(annotations)
    rows = list(read_jsonl(annotations))
    audit = audit_records(rows, image_root=image_root)
    if (audit["errors"] or audit["counts"].get("duplicate_question_ids", 0)
            or any(audit["image_overlap_by_split"].values())
            or any(audit["image_path_overlap_by_split"].values())
            or any(not audit["counts"].get(f"split_{split}")
                   for split in ("train", "validation", "test"))):
        raise ValueError("Require canonical image-disjoint all-split annotations")
    candidates = sorted((r for r in rows if r["split"] == "validation"),
                        key=lambda r: r["question_id"])
    random.Random(seed).shuffle(candidates)
    selected, seen_ids, seen_paths, seen_hashes = [], set(), set(), set()
    for row in candidates:
        validate_selected_records([row], image_root)
        digest = sha256_file(image_root / row["image_path"])
        if (row["image_id"] in seen_ids or row["image_path"] in seen_paths
                or digest in seen_hashes):
            continue
        seen_ids.add(row["image_id"])
        seen_paths.add(row["image_path"])
        seen_hashes.add(digest)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Not enough distinct validation images")
    offset = random.Random(seed).randrange(1, count)
    donors = selected[offset:] + selected[:offset]
    permuted = [{**row, "image_id": donor["image_id"], "image_path": donor["image_path"]}
                for row, donor in zip(selected, donors, strict=True)]
    if sha256_file(annotations) != annotations_hash:
        raise ValueError("Annotations changed during diagnostic selection")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "correct_images.jsonl", selected)
    write_jsonl(output / "permuted_images.jsonl", permuted)
    manifest = {
        "schema": "validation_image_permutation_v1", "count": count, "seed": seed,
        "annotations_sha256": annotations_hash, "offset": offset,
        "files": {name: sha256_file(output / name) for name in (
            "correct_images.jsonl", "permuted_images.jsonl"
        )},
        "image_hashes": {row["image_path"]: sha256_file(image_root / row["image_path"])
                         for row in selected},
        "limitations": "Question-weighted selection, one question per image. Sensitivity "
                       "diagnostic only; changed output does not prove grounded explanations.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_images(bundle, annotations, image_root):
    bundle, annotations, image_root = Path(bundle), Path(annotations), Path(image_root)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    names = ("correct_images.jsonl", "permuted_images.jsonl")
    if (manifest.get("schema") != "validation_image_permutation_v1"
            or manifest.get("annotations_sha256") != sha256_file(annotations)
            or type(manifest.get("count")) is not int or manifest["count"] < 2
            or type(manifest.get("seed")) is not int or manifest["seed"] < 0
            or type(manifest.get("offset")) is not int
            or not 1 <= manifest["offset"] < manifest["count"]
            or set(manifest.get("files", {})) != set(names)):
        raise ValueError("Image diagnostic manifest or canonical annotations changed")
    data = {}
    for name in names:
        if sha256_file(bundle / name) != manifest["files"][name]:
            raise ValueError("Diagnostic annotations changed")
        data[name] = list(read_jsonl(bundle / name))
        validate_selected_records(data[name], image_root)
    original, permuted = (data[name] for name in names)
    canonical = {r["question_id"]: r for r in read_jsonl(annotations)}
    if len(original) != len(permuted) or len(original) != manifest.get("count"):
        raise ValueError("Diagnostic count changed")
    if len({r["image_id"] for r in original}) != len(original):
        raise ValueError("Diagnostic images must be distinct")
    paths = {r["image_path"] for r in original}
    if (len(paths) != len(original) or set(manifest.get("image_hashes", {})) != paths
            or len(set(manifest["image_hashes"].values())) != len(original)):
        raise ValueError("Diagnostic image hashes must cover distinct original images")
    offset = manifest["offset"]
    donors = original[offset:] + original[:offset]
    expected = [{**row, "image_id": donor["image_id"], "image_path": donor["image_path"]}
                for row, donor in zip(original, donors, strict=True)]
    if permuted != expected:
        raise ValueError("Permutation differs from the frozen donor mapping")
    for left, right in zip(original, permuted, strict=True):
        if left["split"] != "validation" or left != canonical.get(left["question_id"]):
            raise ValueError("Diagnostic questions must be canonical validation records")
        for field in left:
            if field not in {"image_id", "image_path"} and left[field] != right.get(field):
                raise ValueError("Permutation changed the question, answer or other metadata")
        if left["image_id"] == right["image_id"] or left["image_path"] == right["image_path"]:
            raise ValueError("Permutation has a fixed point")
    for path, digest in manifest["image_hashes"].items():
        if sha256_file(image_root / path) != digest:
            raise ValueError("Diagnostic image bytes changed")
    return manifest, data


def run_image_check(bundle, annotations, image_root, output_dir, experiment_id, adapter,
                    *, exporter=export_predictions):
    manifest, data = verify_images(bundle, annotations, image_root)
    bundle, output_dir = Path(bundle), Path(output_dir)
    if output_dir.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Keep predictions outside the diagnostic input bundle")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir) + ".lock", timeout=0):
        output_dir.mkdir(exist_ok=True)
        for name in data:
            output = output_dir / name
            args = SimpleNamespace(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
                adapter=Path(adapter) if adapter else None, image_root=Path(image_root),
                annotations=bundle / name, output=output, split="validation", limit=0,
                seed=manifest["seed"], min_pixels=200704, max_pixels=200704,
                max_new_tokens=128, load_in_4bit=True, experiment_id=experiment_id,
                resume=output.with_suffix(".jsonl.manifest.json").exists(),
                export_context={"purpose": "validation_image_permutation",
                                "bundle_sha256": sha256_file(bundle / "manifest.json"),
                                "driver_sha256": sha256_file(Path(__file__))},
            )
            exporter(args)


def score_image_check(bundle, annotations, image_root, output_dir):
    diagnostic, data = verify_images(bundle, annotations, image_root)
    predictions, manifests = [], []
    for name, records in data.items():
        path = Path(output_dir) / name
        rows = list(read_jsonl(path))
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text("utf-8"))
        inputs = manifest.get("inputs", {})
        context = inputs.get("context", {})
        if (context.get("purpose") != "validation_image_permutation"
                or context.get("bundle_sha256") != sha256_file(Path(bundle) / "manifest.json")
                or inputs.get("annotations_sha256") != sha256_file(Path(bundle) / name)
                or inputs.get("split") != "validation" or inputs.get("limit") != 0
                or inputs.get("seed") != diagnostic["seed"]
                or inputs.get("image_hashes") != {
                    r["question_id"]: diagnostic["image_hashes"][r["image_path"]]
                    for r in records
                }):
            raise ValueError("Predictions do not belong to this image diagnostic")
        if len(validate_saved_predictions(rows, records, manifest)) != len(records):
            raise ValueError("Image diagnostic export is incomplete")
        predictions.append({r["question_id"]: r for r in rows})
        manifests.append(manifest)
    left_inputs, right_inputs = (m["inputs"].copy() for m in manifests)
    for inputs in (left_inputs, right_inputs):
        for key in ("annotations_sha256", "selected_sha256", "image_hashes"):
            inputs.pop(key, None)
    if left_inputs != right_inputs or manifests[0]["protocol"] != manifests[1]["protocol"]:
        raise ValueError("The two conditions have different inference identities")
    left, right = predictions
    ids = sorted(left)
    correct = [exact_match(left[q]["prediction_answer"], left[q]["reference_answer"]) for q in ids]
    permuted = [exact_match(right[q]["prediction_answer"], left[q]["reference_answer"])
                for q in ids]
    diff = np.array(correct, dtype=float) - np.array(permuted, dtype=float)
    rng = np.random.default_rng(diagnostic["seed"])
    samples = [float(diff[rng.integers(0, len(ids), size=len(ids))].mean()) for _ in range(2000)]
    return {
        "purpose": "validation_image_permutation", "questions": len(ids),
        "experiment_id": left_inputs["experiment_id"], "seed": diagnostic["seed"],
        "correct_image_em": float(np.mean(correct)),
        "permuted_image_em_against_original_answer": float(np.mean(permuted)),
        "em_drop": float(diff.mean()),
        "em_drop_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "answer_change_rate": sum(normalize_answer(left[q]["prediction_answer"]) !=
                                  normalize_answer(right[q]["prediction_answer"]) for q in ids)
                              / len(ids),
        "limitations": diagnostic["limitations"],
    }
