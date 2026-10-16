"""Editable review records and conversion to validated preference pairs."""

import hashlib
import json
import re
from pathlib import Path

from vqa_explanation.data.formatting import format_target
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.preference.freeze import prepare_exports
from vqa_explanation.preference.schema import (
    AI_TRIAGE_VERDICTS,
    ERROR_TYPES,
    REVIEW_STATES,
    reviewer_type,
)
from vqa_explanation.utils.run_provenance import sha256_file

EDITABLE_FIELDS = (
    "review_status",
    "reviewer",
    "reviewer_type",
    "reviewed_at",
    "answer_assessment",
    "explanation_support",
    "canonical_answer_verified",
    "unsupported_span",
    "image_evidence",
    "proposed_chosen_explanation",
    "proposed_rejected_explanation",
    "error_type",
    "exclusion_reason",
    "rejected_origin_kind",
    "rejected_origin_source",
    "rejected_recipe_sha256",
    "chosen_origin_kind",
    "chosen_origin_source",
    "chosen_recipe_sha256",
)


def review_origin_errors(row: dict) -> list[str]:
    """Use the same authorship rules for review audits and pair exports."""
    errors = []
    try:
        actor = reviewer_type(row)
    except ValueError as error:
        return [str(error)]
    for side in ("chosen", "rejected"):
        kind = row.get(f"{side}_origin_kind")
        kind = None if kind == "" else kind
        source = row.get(f"{side}_origin_source")
        recipe = row.get(f"{side}_recipe_sha256")
        allowed = (
            (None, "manual", "synthetic", "model")
            if side == "rejected"
            else (None, "manual", "synthetic")
        )
        if kind not in allowed:
            errors.append(f"{side}_origin_kind is unsupported")
            continue
        if recipe and kind != "synthetic":
            errors.append(f"{side}_recipe_sha256 requires synthetic origin")
        if side == "rejected":
            text = row.get("proposed_rejected_explanation")
            unchanged = isinstance(text, str) and text.strip() == row.get("candidate_explanation")
            if unchanged:
                if kind not in (None, "model") or source or recipe:
                    errors.append("unchanged rejected must retain the raw M1 model origin")
                continue
            if kind == "model":
                errors.append("edited rejected cannot claim raw M1 model origin")
                continue
        if kind == "synthetic":
            if not isinstance(source, str) or not source.strip():
                errors.append(f"synthetic {side}_origin_source is required")
            if not isinstance(recipe, str) or not re.fullmatch(r"[0-9a-f]{64}", recipe):
                errors.append(f"synthetic {side}_recipe_sha256 must be a lowercase SHA-256")
        if actor == "ai" and row.get("review_status") == "accepted" and kind != "synthetic":
            errors.append(f"AI-written {side} must declare synthetic origin")
    return errors


def blank_notes(inputs, predictions, bundle_sha256):
    return [
        {
            "schema": "candidate_review_v2",
            "source_question_id": row["question_id"],
            "image_path": row["image_path"],
            "bundle_sha256": bundle_sha256,
            "reference_answer": row["answer"],
            "candidate_answer": predictions[row["question_id"]]["prediction_answer"],
            "candidate_explanation": predictions[row["question_id"]]["prediction_explanation"],
            **dict.fromkeys(EDITABLE_FIELDS),
            "review_status": "pending",
        }
        for row in inputs
    ]


def render_editor(
    notes,
    *,
    drafts=None,
    queue=None,
    advisories=None,
    assistance_mode="visible",
):
    """Embed data as JSON, never as executable review text."""
    if assistance_mode not in {"visible", "on_request"}:
        raise ValueError("assistance_mode must be visible or on_request")
    # Old saved notes may predate optional provenance fields. Give the editor
    # those keys without changing the caller's records so import preserves them.
    editable_notes = [{**dict.fromkeys(EDITABLE_FIELDS), **row} for row in notes]
    payload = json.dumps(editable_notes, ensure_ascii=False).replace("<", "\\u003c")
    draft_payload = json.dumps(drafts or [], ensure_ascii=False).replace("<", "\\u003c")
    queue = queue or {
        "schema": "candidate_review_queue_v1",
        "label": "Toàn bộ ứng viên",
        "question_ids": [row["source_question_id"] for row in notes],
    }
    known_ids = {row["source_question_id"] for row in notes}
    queue_ids = queue.get("question_ids")
    if (
        queue.get("schema") != "candidate_review_queue_v1"
        or not isinstance(queue.get("label"), str)
        or not queue["label"].strip()
        or not isinstance(queue_ids, list)
        or not queue_ids
        or len(queue_ids) != len(set(queue_ids))
        or any(qid not in known_ids for qid in queue_ids)
    ):
        raise ValueError("Review queue must contain unique known question IDs")
    queue_payload = json.dumps(queue, ensure_ascii=False).replace("<", "\\u003c")
    advisories = advisories or []
    if not isinstance(advisories, list):
        raise ValueError("Review advisories must be a list")
    advisory_ids = [row.get("source_question_id") for row in advisories if isinstance(row, dict)]
    if (
        len(advisory_ids) != len(advisories)
        or len(advisory_ids) != len(set(advisory_ids))
        or any(qid not in known_ids for qid in advisory_ids)
        or any(row.get("ai_verdict") not in AI_TRIAGE_VERDICTS for row in advisories)
    ):
        raise ValueError("Review advisories must contain unique known IDs and valid verdicts")
    advisory_payload = json.dumps(advisories, ensure_ascii=False).replace("<", "\\u003c")
    storage_key = (
        "vqa-review-draft-v1:"
        + hashlib.sha256(
            (payload + "\n" + queue_payload + "\n" + assistance_mode).encode("utf-8")
        ).hexdigest()
    )
    legacy_keys = [
        "vqa-review-draft-v1:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in (
            payload + "\n" + queue_payload,
            payload
            + "\n"
            + queue_payload
            + "\n"
            + ("visible" if assistance_mode == "on_request" else "on_request"),
        )
    ]
    assets = Path(__file__).with_name("templates")
    template = (assets / "editor.html").read_text(encoding="utf-8")
    template = re.sub(
        r"\{\{(stylesheet|javascript)\}\}",
        lambda match: (assets / ("editor.css" if match[1] == "stylesheet" else "editor.js"))
        .read_text(encoding="utf-8"),
        template,
    )
    values = {
        "payload": payload, "draft_payload": draft_payload,
        "queue_payload": queue_payload, "advisory_payload": advisory_payload,
        "assistance_mode": assistance_mode, "storage_key": storage_key,
        "legacy_payload": json.dumps(legacy_keys),
    }
    return re.sub(r"\{\{([a-z_]+)\}\}", lambda match: values[match[1]], template)


def convert_notes(notes, inputs, predictions, manifest, bundle_sha256, annotations, image_root):
    """Require explicit attributed decisions; pending rows never become train data."""
    if manifest["inputs"].get("context", {}).get("annotations_sha256") != sha256_file(annotations):
        raise ValueError("Candidate bundle belongs to a different canonical annotation version")
    expected = {
        row["source_question_id"]: row for row in blank_notes(inputs, predictions, bundle_sha256)
    }
    by_id = {row["question_id"]: row for row in inputs}
    immutable = set(next(iter(expected.values()))) - set(EDITABLE_FIELDS)
    if len(notes) != len(expected):
        raise ValueError("Review must retain every candidate, including excluded rows")
    seen, pairs = set(), []
    recipe_hash = hashlib.sha256(
        json.dumps(
            {"inputs": manifest["inputs"], "protocol": manifest["protocol"]},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    adapter_files = manifest["inputs"].get("adapter_files")
    if not adapter_files:
        raise ValueError("Candidate manifest must identify the M1 adapter files")
    candidate_origin = {
        "model_id": manifest["inputs"]["model_id"],
        "checkpoint_sha256": hashlib.sha256(
            json.dumps(adapter_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    for note in notes:
        qid = note.get("source_question_id")
        if qid in seen or qid not in expected:
            raise ValueError("Unknown or duplicate review ID")
        seen.add(qid)
        if any(note.get(key) != expected[qid][key] for key in immutable):
            raise ValueError(f"{qid}: immutable review source changed")
        if note.get("review_status") not in REVIEW_STATES:
            raise ValueError(f"{qid}: invalid review_status")
        if note["review_status"] != "accepted":
            continue
        actor = reviewer_type(note)
        if type(note.get("explanation_support")) is not int or note["explanation_support"] not in {
            0,
            1,
            2,
        }:
            raise ValueError(f"{qid}: record explanation_support as 0, 1 or 2")
        if (
            note.get("answer_assessment") != "correct"
            or note.get("canonical_answer_verified") is not True
        ):
            raise ValueError(f"{qid}: verify both candidate and canonical answers from the image")
        if note.get("error_type") not in ERROR_TYPES:
            raise ValueError(f"{qid}: choose an error_type")
        for key in (
            "proposed_chosen_explanation",
            "proposed_rejected_explanation",
            "unsupported_span",
            "image_evidence",
            "reviewer",
            "reviewed_at",
        ):
            if not isinstance(note.get(key), str) or not note[key].strip():
                raise ValueError(f"{qid}: {key} is required")
        row = by_id[qid]
        original = predictions[qid]["prediction_explanation"]
        rejected = note["proposed_rejected_explanation"].strip()
        origin_errors = review_origin_errors(note)
        if origin_errors:
            raise ValueError(f"{qid}: {'; '.join(origin_errors)}")
        chosen_kind = note.get("chosen_origin_kind")
        if chosen_kind == "":
            chosen_kind = None
        chosen_source = note.get("chosen_origin_source")
        chosen_recipe = note.get("chosen_recipe_sha256")
        chosen_origin = {
            "kind": "manual",
            "source": (
                chosen_source.strip()
                if isinstance(chosen_source, str) and chosen_source.strip()
                else f"review:{qid}"
            ),
        }
        if chosen_kind == "synthetic":
            chosen_origin = {
                "kind": "synthetic",
                "source": chosen_source.strip(),
                "recipe_sha256": chosen_recipe,
            }
        origin_kind = note.get("rejected_origin_kind")
        if origin_kind == "":
            origin_kind = None
        source = note.get("rejected_origin_source")
        recipe = note.get("rejected_recipe_sha256")
        origin = {
            "kind": "manual",
            "source": (
                source.strip()
                if isinstance(source, str) and source.strip()
                else f"human review of {bundle_sha256}:{qid}"
            ),
        }
        if rejected == original:
            if not adapter_files:
                raise ValueError("Candidate manifest must identify the M1 adapter files")
            origin = {
                "kind": "model",
                "source": f"{bundle_sha256}:candidates.jsonl:{qid}",
                "model_id": manifest["inputs"]["model_id"],
                "checkpoint_sha256": hashlib.sha256(
                    json.dumps(adapter_files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "recipe_sha256": recipe_hash,
                "seed": manifest["inputs"]["seed"],
            }
        elif origin_kind == "synthetic":
            origin = {"kind": "synthetic", "source": source.strip(), "recipe_sha256": recipe}
        pairs.append(
            {
                "pair_id": f"review-{bundle_sha256[:12]}-{qid}",
                "source_question_id": qid,
                "image_path": row["image_path"],
                "prompt": build_user_prompt(row["question"]),
                "chosen": format_target(row["answer"], note["proposed_chosen_explanation"].strip()),
                "rejected": format_target(row["answer"], rejected),
                "error_type": note["error_type"],
                "generation_method": f"{actor}_review_candidate",
                "review_status": "accepted",
                "reviewer": note["reviewer"],
                "reviewer_type": actor,
                "human_verified": actor == "human",
                "reviewed_at": note["reviewed_at"],
                "evidence": note["image_evidence"],
                "rejected_span": note["unsupported_span"],
                "annotations_sha256": sha256_file(annotations),
                "image_sha256": manifest["inputs"]["image_hashes"][qid],
                "provenance": {
                    "chosen": chosen_origin,
                    "rejected": origin,
                    "review_bundle_sha256": bundle_sha256,
                    "original_candidate_origin": candidate_origin,
                    "original_candidate_answer": note["candidate_answer"],
                    "exported_answer": row["answer"],
                    "rejected_explanation_edited": rejected != original,
                },
            }
        )
    if not pairs:
        raise ValueError("No accepted pairs; explicit review is required")
    canonical = list(read_jsonl(annotations))
    accepted, _, _ = prepare_exports(pairs, canonical, image_root, sha256_file(annotations))
    return accepted


def export_review(
    notes_path, inputs, predictions, manifest, bundle_sha256, annotations, image_root, output_dir
):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review export: {output}")
    notes = list(read_jsonl(notes_path))
    pairs = convert_notes(
        notes, inputs, predictions, manifest, bundle_sha256, annotations, image_root
    )
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "reviewed_pairs.jsonl", pairs)
    write_json(
        output / "review_summary.json",
        {
            "review_notes_sha256": sha256_file(notes_path),
            "bundle_sha256": bundle_sha256,
            "review_counts": {
                state: sum(n["review_status"] == state for n in notes)
                for state in sorted(REVIEW_STATES)
            },
            "accepted": len(pairs),
            "reviewer_types": {
                actor: sum(p["reviewer_type"] == actor for p in pairs)
                for actor in ("human", "ai")
            },
            "status": "validated_metadata",
            "limitations": "Does not verify reviewer identity or visual truth.",
        },
    )
    return output / "reviewed_pairs.jsonl"
