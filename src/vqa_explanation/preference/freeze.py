"""Link reviewed pairs to canonical train and freeze matched M2/M3 inputs."""

import re
import unicodedata
from collections import Counter
from pathlib import Path

from vqa_explanation.data.formatting import format_target
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.data.schema import VQASample
from vqa_explanation.inference.prompting import build_user_prompt, parse_prediction
from vqa_explanation.preference.schema import PreferencePair, reviewer_type
from vqa_explanation.preference.validate import validate_pairs
from vqa_explanation.utils.review_time import parse_review_time
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _image_path(root, relative):
    # Relative paths keep exports portable.
    if (not relative or "\\" in relative or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError(f"Noncanonical image path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes root: {relative}")
    return path


def prepare_exports(pairs, canonical, image_root, annotations_sha256):
    """Validate pair metadata, not visual quality or reviewer identity."""
    report = validate_pairs(pairs)
    if report["errors"]:
        raise ValueError(f"Invalid pair schema: {report['errors']}")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise ValueError("image_root must be a directory")
    samples = {}
    image_splits = {}
    path_splits = {}
    for raw in canonical:
        sample = VQASample.from_dict(raw)
        if sample.question_id in samples:
            raise ValueError(f"Duplicate canonical question: {sample.question_id}")
        samples[sample.question_id] = sample
        path = _image_path(root, sample.image_path)
        for mapping, key in ((image_splits, sample.image_id), (path_splits, path)):
            if key in mapping and mapping[key] != sample.split:
                raise ValueError(f"Canonical image crosses splits: {sample.image_path}")
            mapping[key] = sample.split

    accepted, chosen_rows = [], []
    seen_content, seen_questions = set(), set()
    hashes = {}
    for raw in pairs:
        pair = PreferencePair.from_dict(raw)
        sample = samples.get(pair.source_question_id)
        if sample is None or sample.split != "train":
            raise ValueError(f"{pair.pair_id}: source question must belong to canonical train")
        if pair.image_path != sample.image_path:
            raise ValueError(f"{pair.pair_id}: image_path differs from canonical train")
        if pair.prompt != build_user_prompt(sample.question):
            raise ValueError(f"{pair.pair_id}: prompt differs from canonical question/template")
        if pair.review_status != "accepted":
            continue
        actor = reviewer_type(raw)
        if actor == "ai" and raw.get("human_verified") is not False:
            raise ValueError(f"{pair.pair_id}: AI review requires human_verified=false")
        if "human_verified" in raw and raw["human_verified"] is not (actor == "human"):
            raise ValueError(f"{pair.pair_id}: human_verified differs from reviewer_type")
        if pair.generation_method in {"ai_review_candidate", "human_review_candidate"}:
            if pair.generation_method != f"{actor}_review_candidate":
                raise ValueError(f"{pair.pair_id}: review generation_method differs from actor")
        if sample.question_id in seen_questions:
            raise ValueError(f"Duplicate accepted source question: {sample.question_id}")
        seen_questions.add(sample.question_id)
        if raw.get("annotations_sha256") != annotations_sha256:
            raise ValueError(f"{pair.pair_id}: annotations_sha256 mismatch")
        image_path = _image_path(root, sample.image_path)
        if image_path not in hashes:
            hashes[image_path] = sha256_file(image_path)
        if _hash(raw.get("image_sha256"), "image_sha256") != hashes[image_path]:
            raise ValueError(f"{pair.pair_id}: image_sha256 mismatch")
        for key in ("reviewer", "evidence", "rejected_span"):
            _text(raw.get(key), key)
        parse_review_time(_text(raw.get("reviewed_at"), "reviewed_at"))
        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        for side in ("chosen", "rejected"):
            origin = provenance.get(side)
            if not isinstance(origin, dict):
                raise ValueError(f"provenance.{side} must be an object")
            if not isinstance(origin.get("kind"), str) or origin["kind"] not in {
                "reference", "manual", "model", "synthetic"
            }:
                raise ValueError(f"provenance.{side}.kind is unsupported")
            _text(origin.get("source"), f"provenance.{side}.source")
            if actor == "ai" and origin["kind"] == "manual":
                raise ValueError(f"{pair.pair_id}: AI-written text cannot claim manual origin")
            if origin["kind"] in {"model", "synthetic"}:
                _hash(origin.get("recipe_sha256"), f"provenance.{side}.recipe_sha256")
            if origin["kind"] == "model":
                _text(origin.get("model_id"), f"provenance.{side}.model_id")
                _hash(origin.get("checkpoint_sha256"), "checkpoint_sha256")
                if type(origin.get("seed")) is not int or origin["seed"] < 0:
                    raise ValueError("Model generation seed must be a non-negative integer")
        chosen, rejected = parse_prediction(pair.chosen), parse_prediction(pair.rejected)
        if not chosen.format_valid or not rejected.format_valid:
            raise ValueError(f"{pair.pair_id}: both outputs must use answer/explanation format")
        if not (_normalized(chosen.answer) == _normalized(rejected.answer)
                == _normalized(sample.answer)):
            raise ValueError(f"{pair.pair_id}: answers differ from each other or canonical answer")
        if _normalized(chosen.explanation) == _normalized(rejected.explanation):
            raise ValueError(f"{pair.pair_id}: explanations must differ")
        if _normalized(raw["rejected_span"]) not in _normalized(rejected.explanation):
            raise ValueError(f"{pair.pair_id}: rejected_span is absent from rejected explanation")
        content = (hashes[image_path], _normalized(sample.question),
                   _normalized(sample.answer), _normalized(chosen.explanation),
                   _normalized(rejected.explanation))
        if content in seen_content:
            raise ValueError(f"{pair.pair_id}: duplicate accepted content")
        seen_content.add(content)
        # M2 and M3 use the same answer format.
        chosen_text = format_target(sample.answer, chosen.explanation)
        accepted.append({**raw, "pair_id": pair.pair_id, "split": "train",
                         "source_question_id": sample.question_id,
                         "image_path": sample.image_path,
                         "image_id": sample.image_id, "question": sample.question,
                         "prompt": build_user_prompt(sample.question),
                         "chosen": chosen_text,
                         "rejected": format_target(sample.answer, rejected.explanation)})
        chosen_rows.append({"question_id": sample.question_id, "image_id": sample.image_id,
                            "image_path": sample.image_path, "question": sample.question,
                            "answer": sample.answer, "explanations": [chosen.explanation],
                            "split": "train", "pair_id": pair.pair_id})
    if not accepted:
        raise ValueError("No accepted pairs; explicit review is required before training export")
    accepted.sort(key=lambda row: row["pair_id"])
    chosen_rows.sort(key=lambda row: row["pair_id"])
    return accepted, chosen_rows, report


def freeze_pairs(input_path, annotations, image_root, output_dir):
    """Validate inputs and write a new bundle without overwriting existing files."""
    input_path, annotations, output_dir = map(Path, (input_path, annotations, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite preference bundle: {output_dir}")
    input_hash, annotations_hash = sha256_file(input_path), sha256_file(annotations)
    pairs, chosen, report = prepare_exports(
        list(read_jsonl(input_path)), list(read_jsonl(annotations)), image_root, annotations_hash
    )
    if input_hash != sha256_file(input_path) or annotations_hash != sha256_file(annotations):
        raise ValueError("Input files changed during validation")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "accepted.jsonl", pairs)
    write_jsonl(output_dir / "chosen_only.jsonl", chosen)
    manifest = {
        "schema_version": 1, "status": "validated_metadata", "created_at_utc": utc_now(),
        "input_sha256": input_hash, "annotations_sha256": annotations_hash,
        "pairs": len(pairs), "unique_images": len({p["image_sha256"] for p in pairs}),
        "review_counts": report["counts"],
        "error_types": dict(Counter(p["error_type"] for p in pairs)),
        "rejected_sources": dict(Counter(p["provenance"]["rejected"]["kind"] for p in pairs)),
        "reviewer_types": dict(Counter(reviewer_type(p) for p in pairs)),
        "human_verified_pairs": sum(reviewer_type(p) == "human" for p in pairs),
        "chosen_explanation_words": [len(r["explanations"][0].split()) for r in chosen],
        "rejected_explanation_words": [len(parse_prediction(p["rejected"]).explanation.split())
                                       for p in pairs],
        "files": {name: sha256_file(output_dir / name)
                  for name in ("accepted.jsonl", "chosen_only.jsonl")},
        "limitations": "Validates supplied review metadata, not reviewer identity or visual truth. "
                        "Canonical data must also pass the dataset/image-disjoint audit.",
    }
    # Write the manifest last to mark the bundle as complete.
    write_json(output_dir / "manifest.json", manifest)
    return manifest
