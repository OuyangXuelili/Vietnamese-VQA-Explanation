"""Verify AI proposals before showing them alongside pending review records."""

import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.preference.review import EDITABLE_FIELDS, blank_notes
from vqa_explanation.preference.schema import ERROR_TYPES
from vqa_explanation.utils.run_provenance import sha256_file


def load_review_drafts(review_dir, drafts_path, recipe_path):
    root = Path(review_dir).resolve()
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    recipe_hash = sha256_file(recipe_path)
    for key, name in (
        ("source_inputs", "inputs.jsonl"),
        ("source_candidates", "candidates.jsonl"),
        ("source_candidate_manifest", "candidates.jsonl.manifest.json"),
    ):
        if sha256_file(root / name) != recipe[key]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    inputs = list(read_jsonl(root / "inputs.jsonl"))
    predictions = {row["question_id"]: row for row in read_jsonl(root / "candidates.jsonl")}
    manifest = json.loads((root / "candidates.jsonl.manifest.json").read_text(encoding="utf-8"))
    if manifest["inputs"]["context"]["annotations_sha256"] != recipe["annotations_sha256"]:
        raise ValueError("Canonical annotation checksum mismatch")
    notes = list(read_jsonl(root / "review_notes.jsonl"))
    expected = blank_notes(inputs, predictions, recipe["source_bundle_sha256"])
    immutable = set(expected[0]) - set(EDITABLE_FIELDS)
    if len(notes) != len(expected) or any(
        any(note.get(key) != base[key] for key in immutable)
        for note, base in zip(notes, expected, strict=True)
    ):
        raise ValueError("Review records do not match the candidate source/order")
    by_id = {row["question_id"]: row for row in inputs}
    drafts = list(read_jsonl(drafts_path))
    ids = [row["source_question_id"] for row in drafts]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(recipe["selection"]["question_ids"]):
        raise ValueError("Draft IDs must match the recipe without duplicates")
    transforms = {row["source_question_id"]: row for row in recipe["transformations"]}
    for draft in drafts:
        qid = draft["source_question_id"]
        if qid not in by_id:
            raise ValueError(f"{qid}: unknown draft ID")
        row, candidate = by_id[qid], predictions[qid]
        if (
            draft.get("schema") != "preference_ai_draft_v1"
            or draft.get("status") != "pending_ai_draft"
            or draft.get("human_verified") is not False
            or draft.get("reviewer") is not None
            or draft.get("reviewed_at") is not None
        ):
            raise ValueError(f"{qid}: drafts must remain unreviewed AI proposals")
        matches = (
            row["split"] == draft["split"] == "train",
            draft["question"] == row["question"],
            draft["answer"] == row["answer"] == candidate["prediction_answer"],
            draft["original_candidate_answer"] == candidate["prediction_answer"],
            draft["original_candidate_explanation"] == candidate["prediction_explanation"],
            draft["canonical_image_path"] == row["image_path"],
            draft["recipe_sha256"] == recipe_hash,
            draft["annotations_sha256"] == recipe["annotations_sha256"],
            draft["source_candidate"]["sha256"] == recipe["source_candidates"]["sha256"],
            draft["source_candidate"]["question_id"] == qid,
        )
        if not all(matches):
            raise ValueError(f"{qid}: draft source mismatch")
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"{qid}: image escapes review directory")
        if sha256_file(image) != draft["image_sha256"] or (
            draft["image_sha256"] != manifest["inputs"]["image_hashes"][qid]
        ):
            raise ValueError(f"{qid}: image checksum mismatch")
        for field in (
            "proposed_chosen_explanation", "proposed_rejected_explanation",
            "unsupported_span", "image_evidence_ai",
        ):
            if not isinstance(draft.get(field), str) or not draft[field].strip():
                raise ValueError(f"{qid}: missing {field}")
        chosen, rejected = (
            draft[f"proposed_{side}_explanation"] for side in ("chosen", "rejected")
        )
        if chosen == rejected or draft["unsupported_span"] not in rejected:
            raise ValueError(f"{qid}: invalid contrast or rejected span")
        if draft["error_type"] not in ERROR_TYPES:
            raise ValueError(f"{qid}: invalid error type")
        sides = ["chosen"]
        if draft["origin_kind"] == "synthetic":
            sides.append("rejected")
            transform = transforms[qid]
            if (
                chosen != transform["chosen_explanation"]
                or not transform["replace_from"]
                or chosen.count(transform["replace_from"]) != 1
                or rejected != chosen.replace(transform["replace_from"], transform["replace_to"], 1)
                or draft["unsupported_span"] != transform["replace_to"]
                or draft["error_type"] != transform["error_type"]
            ):
                raise ValueError(f"{qid}: synthetic edit differs from recipe")
        elif draft["origin_kind"] == "model":
            if rejected != candidate["prediction_explanation"]:
                raise ValueError(f"{qid}: model rejected must retain original text")
        else:
            raise ValueError(f"{qid}: unsupported rejected origin")
        for side in sides:
            origin = draft[f"{side}_provenance"]
            if (
                origin["kind"] != "synthetic" or origin["recipe_sha256"] != recipe_hash
                or not isinstance(origin["source"], str) or not origin["source"].strip()
            ):
                raise ValueError(f"{qid}: invalid synthetic provenance")
    return notes, drafts
