import copy

from vqa_explanation.preference.audit import audit_review_notes
from vqa_explanation.preference.review import EDITABLE_FIELDS


def baseline_note(question_id: str = "q1") -> dict:
    return {
        "schema": "candidate_review_v2",
        "source_question_id": question_id,
        "image_path": f"images/{question_id}.jpg",
        "bundle_sha256": "b" * 64,
        "reference_answer": "cat",
        "candidate_answer": "cat",
        "candidate_explanation": "It has wings.",
        **dict.fromkeys(EDITABLE_FIELDS),
        "review_status": "pending",
    }


def accepted_note() -> dict:
    row = baseline_note()
    row.update(
        review_status="accepted",
        reviewer="reviewer one",
        reviewed_at="2026-10-10T12:00:00+07:00",
        answer_assessment="correct",
        explanation_support=0,
        canonical_answer_verified=True,
        unsupported_span="wings",
        image_evidence="The animal has visible fur and no wings.",
        proposed_chosen_explanation="It has fur.",
        proposed_rejected_explanation="It has wings.",
        error_type="attribute",
    )
    return row


def test_pending_ai_draft_is_reported_but_not_failed() -> None:
    baseline = baseline_note()
    draft = copy.deepcopy(baseline)
    draft["proposed_chosen_explanation"] = "It has fur."

    report = audit_review_notes([draft], [baseline], ["q1"])

    assert report["status"] == "passed"
    assert report["counts"] == {"draft_pending": 1, "pending": 1, "records": 1}
    assert report["pilot_counts"] == {"draft_pending": 1, "pending": 1, "records": 1}


def test_complete_accepted_review_passes_readiness_audit() -> None:
    baseline = baseline_note()

    report = audit_review_notes([accepted_note()], [baseline], ["q1"])

    assert report["status"] == "passed"
    assert report["counts"]["accepted"] == 1
    assert report["errors"] == []


def test_resolved_exclusion_requires_reason_and_reviewer() -> None:
    baseline = baseline_note()
    rejected = copy.deepcopy(baseline)
    rejected["review_status"] = "rejected"

    report = audit_review_notes([rejected], [baseline])

    assert report["status"] == "failed"
    assert report["counts"]["invalid"] == 1
    assert "reviewer is required" in " ".join(report["errors"][0]["errors"])
    assert "exclusion_reason is required" in " ".join(report["errors"][0]["errors"])


def test_forged_source_and_bad_synthetic_origin_are_rejected() -> None:
    baseline = baseline_note()
    row = accepted_note()
    row.update(
        candidate_explanation="forged",
        chosen_origin_kind="synthetic",
        chosen_origin_source=None,
        chosen_recipe_sha256="bad",
    )

    report = audit_review_notes([row], [baseline])
    messages = " ".join(report["errors"][0]["errors"])

    assert report["status"] == "failed"
    assert "immutable review source changed" in messages
    assert "chosen_origin_source" in messages
    assert "chosen_recipe_sha256" in messages
