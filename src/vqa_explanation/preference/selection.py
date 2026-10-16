"""Freeze a train question pool for preference annotation."""

import json
import random
from collections import Counter
from pathlib import Path

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.data.schema import VQASample
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def select_questions(records, image_root, count=2000, seed=42, excluded=()):
    if type(count) is not int or not 1 <= count <= 2000:
        raise ValueError("count must be an integer between 1 and 2000")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise ValueError("image_root must be a directory")
    records = list(records)
    audit = audit_records(records, image_root=root)
    if (audit["errors"] or audit["counts"].get("duplicate_question_ids", 0)
            or any(audit["image_overlap_by_split"].values())
            or any(audit["image_path_overlap_by_split"].values())):
        raise ValueError("Canonical annotations failed the ID/split audit")
    if any(not audit["counts"].get(f"split_{split}")
           for split in ("train", "validation", "test")):
        raise ValueError("Supply canonical all.jsonl containing all three splits")
    by_id = {str(row["question_id"]): row for row in records}
    excluded_ids, excluded_images, excluded_paths = set(), set(), set()
    for row in excluded:
        original = by_id.get(row.get("source_question_id"))
        if not original or original["split"] != "train" or row.get("split") != "train":
            raise ValueError("Exclusions must contain canonical train questions")
        if any(row.get(key) != original[key] for key in ("image_id", "image_path")):
            raise ValueError("Excluded image differs from canonical annotations")
        excluded_ids.add(row["source_question_id"])
        excluded_images.add(row["image_id"])
        excluded_paths.add((root / row["image_path"]).resolve())
    candidates = []
    for raw in records:
        sample = VQASample.from_dict(raw)
        relative = sample.image_path
        if ("\\" in relative or ":" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or not (root / relative).resolve().is_relative_to(root)):
            raise ValueError(f"Noncanonical image path: {relative}")
        if raw["split"] != sample.split:
            raise ValueError("Noncanonical split label")
        if sample.split == "train":
            candidates.append((sample, raw))
    candidates.sort(key=lambda item: item[0].question_id)
    random.Random(seed).shuffle(candidates)
    selected = []
    seen_ids, seen_paths = set(), set()
    for sample, raw in candidates:
        path = (root / sample.image_path).resolve()
        if (sample.question_id in excluded_ids or sample.image_id in excluded_images
                or path in excluded_paths):
            continue
        if sample.image_id in seen_ids or path in seen_paths:
            continue
        seen_ids.add(sample.image_id)
        seen_paths.add(path)
        selected.append({
            "source_question_id": sample.question_id, "image_id": sample.image_id,
            "image_path": sample.image_path, "split": "train", "question": raw["question"],
            "reference_answer": raw["answer"],
            "reference_explanations": list(raw["explanations"]),
            "question_type": raw.get("question_type"),
            "prompt": build_user_prompt(sample.question), "review_status": "pending",
        })
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} distinct train images; requested {count}")
    return selected


def freeze_selection(annotations, image_root, output_dir, count=2000, seed=42,
                     exclude_selections=()):
    annotations, root, output = Path(annotations), Path(image_root), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite question selection: {output}")
    digest = sha256_file(annotations)
    excluded, exclusion_hashes = [], {}
    for directory in exclude_selections:
        directory = Path(directory)
        path = directory / "questions.jsonl"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_hash = sha256_file(path)
        if (manifest.get("status") != "questions_selected_not_reviewed"
                or manifest.get("schema_version") != 1
                or manifest.get("annotations_sha256") != digest
                or manifest.get("files", {}).get("questions.jsonl") != file_hash):
            raise ValueError("Excluded selection manifest/hash changed")
        rows = list(read_jsonl(path))
        if len(rows) != manifest.get("count"):
            raise ValueError("Excluded selection count changed")
        excluded.extend(rows)
        exclusion_hashes[str(path.resolve())] = file_hash
        exclusion_hashes[str(manifest_path.resolve())] = sha256_file(manifest_path)
    selected = select_questions(read_jsonl(annotations), root, count, seed, excluded)
    for row in selected:
        row["annotations_sha256"] = digest
        row["image_sha256"] = sha256_file(root / row["image_path"])
    if sha256_file(annotations) != digest:
        raise ValueError("Annotations changed during selection")
    if any(sha256_file(Path(path)) != expected for path, expected in exclusion_hashes.items()):
        raise ValueError("Excluded selection changed during selection")
    output.mkdir(parents=True, exist_ok=False)
    questions = output / "questions.jsonl"
    write_jsonl(questions, selected)
    manifest = {
        "schema_version": 1, "status": "questions_selected_not_reviewed",
        "created_at_utc": utc_now(), "annotations_sha256": digest,
        "seed": seed, "count": len(selected),
        "method": "sort question IDs, seeded shuffle, first question per image ID/path",
        "calibration_question_ids": [r["source_question_id"] for r in selected[:100]],
        "question_types": dict(Counter(str(r["question_type"]) for r in selected)),
        "files": {questions.name: sha256_file(questions)},
        "limitations": "Not a preference-pair or training export. References are unreviewed. "
                        "Sampling is question-weighted, not uniform over images or types. "
                        "One-per-image uses canonical IDs/paths, not perceptual deduplication.",
    }
    if exclusion_hashes:
        manifest["excluded_selection_files"] = exclusion_hashes
        manifest["excluded_images"] = len({row["image_id"] for row in excluded})
    write_json(output / "manifest.json", manifest)
    return manifest
