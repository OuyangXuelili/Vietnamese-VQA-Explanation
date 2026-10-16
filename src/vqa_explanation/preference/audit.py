"""Audit partial review exports before conversion to training pairs."""

from __future__ import annotations

import unicodedata
from collections import Counter

from vqa_explanation.preference.review import EDITABLE_FIELDS, review_origin_errors
from vqa_explanation.preference.schema import ERROR_TYPES, REVIEW_STATES, reviewer_type
from vqa_explanation.utils.review_time import parse_review_time


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _valid_time(value: object) -> bool:
    if not _text(value):
        return False
    try:
        parse_review_time(str(value))
    except ValueError:
        return False
    return True


def _accepted_errors(row: dict) -> list[str]:
    errors = []
    if row.get("answer_assessment") != "correct":
        errors.append("answer_assessment must be correct")
    if row.get("canonical_answer_verified") is not True:
        errors.append("canonical_answer_verified must be true")
    if type(row.get("explanation_support")) is not int or row["explanation_support"] not in {
        0,
        1,
        2,
    }:
        errors.append("explanation_support must be 0, 1 or 2")
    if row.get("error_type") not in ERROR_TYPES:
        errors.append("error_type is required")
    for field in (
        "unsupported_span",
        "image_evidence",
        "proposed_chosen_explanation",
        "proposed_rejected_explanation",
    ):
        if not _text(row.get(field)):
            errors.append(f"{field} is required")
    chosen = _normalized(row.get("proposed_chosen_explanation"))
    rejected = _normalized(row.get("proposed_rejected_explanation"))
    span = _normalized(row.get("unsupported_span"))
    if chosen and rejected and chosen == rejected:
        errors.append("chosen and rejected explanations must differ")
    if span and rejected and span not in rejected:
        errors.append("unsupported_span is absent from rejected explanation")
    errors.extend(review_origin_errors(row))
    return errors


def audit_review_notes(
    notes: list[dict], baseline: list[dict], pilot_ids: list[str] | None = None
) -> dict:
    """Check source integrity and readiness without creating accepted pairs."""
    counts = Counter(records=len(notes))
    errors = []
    baseline_by_id = {row.get("source_question_id"): row for row in baseline}
    if len(baseline_by_id) != len(baseline):
        errors.append({"source_question_id": None, "errors": ["baseline contains duplicate IDs"]})
    if len(notes) != len(baseline):
        errors.append(
            {"source_question_id": None, "errors": ["review row count differs from baseline"]}
        )
    immutable = set(baseline[0]) - set(EDITABLE_FIELDS) if baseline else set()
    seen = set()
    pilot_set = set(pilot_ids or [])
    pilot_counts = Counter(records=0)

    for row in notes:
        qid = row.get("source_question_id")
        row_errors = []
        if qid in seen:
            row_errors.append("duplicate source_question_id")
        seen.add(qid)
        expected = baseline_by_id.get(qid)
        if expected is None:
            row_errors.append("unknown source_question_id")
        elif any(row.get(key) != expected.get(key) for key in immutable):
            row_errors.append("immutable review source changed")

        status = row.get("review_status")
        if status not in REVIEW_STATES:
            row_errors.append("invalid review_status")
        else:
            counts[status] += 1
            if qid in pilot_set:
                pilot_counts[status] += 1
                pilot_counts["records"] += 1

        edited = any(
            row.get(key) not in (None, "", "pending")
            for key in EDITABLE_FIELDS
            if key != "review_status"
        )
        if status == "pending":
            if edited:
                counts["draft_pending"] += 1
                if qid in pilot_set:
                    pilot_counts["draft_pending"] += 1
        elif status in REVIEW_STATES:
            try:
                reviewer_type(row)
            except ValueError as error:
                row_errors.append(str(error))
            if not _text(row.get("reviewer")):
                row_errors.append("reviewer is required for a resolved decision")
            if not _valid_time(row.get("reviewed_at")):
                row_errors.append("reviewed_at must include a timezone")
            if status == "accepted":
                row_errors.extend(_accepted_errors(row))
            elif not _text(row.get("exclusion_reason")):
                row_errors.append("exclusion_reason is required when not accepted")

        if row_errors:
            counts["invalid"] += 1
            if qid in pilot_set:
                pilot_counts["invalid"] += 1
            errors.append({"source_question_id": qid, "errors": row_errors})

    missing = set(baseline_by_id) - seen
    if missing:
        errors.append(
            {
                "source_question_id": None,
                "errors": [f"missing {len(missing)} baseline question IDs"],
            }
        )
    if pilot_ids is not None:
        if len(pilot_ids) != len(pilot_set) or not pilot_set.issubset(baseline_by_id):
            errors.append(
                {"source_question_id": None, "errors": ["pilot IDs are invalid or unknown"]}
            )
        missing_pilot = pilot_set - seen
        if missing_pilot:
            errors.append(
                {
                    "source_question_id": None,
                    "errors": [f"missing {len(missing_pilot)} pilot question IDs"],
                }
            )

    return {
        "schema": "preference_review_audit_v1",
        "status": "passed" if not errors else "failed",
        "counts": dict(sorted(counts.items())),
        "pilot_counts": dict(sorted(pilot_counts.items())) if pilot_ids is not None else None,
        "errors": errors,
        "limitations": (
            "Checks source integrity and field completeness; it does not verify reviewer identity "
            "or visual truth."
        ),
    }
