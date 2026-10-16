import copy

import pytest

from vqa_explanation.preference.pilot_report import (
    render_pilot_markdown,
    summarize_pilot,
    wilson_interval,
)
from vqa_explanation.preference.review import EDITABLE_FIELDS


def baseline(question_id: str) -> dict:
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


def resolved(row: dict, status: str) -> dict:
    item = copy.deepcopy(row)
    item.update(
        review_status=status,
        reviewer="reviewer one",
        reviewed_at="2026-10-10T12:00:00+07:00",
    )
    if status == "accepted":
        item.update(
            answer_assessment="correct",
            explanation_support=0,
            canonical_answer_verified=True,
            unsupported_span="wings",
            image_evidence="The cat has fur and no wings.",
            proposed_chosen_explanation="It has fur.",
            proposed_rejected_explanation="It has wings.",
            error_type="attribute",
        )
    else:
        item["exclusion_reason"] = "No usable contrastive pair."
    return item


def test_wilson_interval_matches_known_half_rate() -> None:
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=0.0001)
    assert high == pytest.approx(0.5962, abs=0.0001)


def test_pending_pilot_does_not_publish_a_yield_or_open_training() -> None:
    rows = [baseline("q1"), baseline("q2")]
    report = summarize_pilot(rows, rows, ["q1", "q2"], target_pairs=10)

    assert report["status"] == "in_progress"
    assert report["acceptance"] is None
    assert report["candidate_pool_projection"] is None
    assert report["gates"]["training_gate_open"] is False
    assert report["gates"]["next_action"] == "finish_fixed_pilot"
    assert "Chưa tính tỷ lệ giữ lại" in render_pilot_markdown(report)


def test_complete_pilot_reports_yield_projection_and_ai_contingency() -> None:
    rows = [baseline(f"q{i}") for i in range(1, 5)]
    notes = [
        resolved(rows[0], "accepted"),
        resolved(rows[1], "accepted"),
        resolved(rows[2], "rejected"),
        resolved(rows[3], "uncertain"),
    ]
    triage = [
        {"source_question_id": "q1", "ai_verdict": "repair_explanation"},
        {"source_question_id": "q2", "ai_verdict": "repair_explanation"},
        {"source_question_id": "q3", "ai_verdict": "pass_grounded"},
        {"source_question_id": "q4", "ai_verdict": "exclude_ambiguous"},
    ]

    report = summarize_pilot(
        notes,
        rows,
        ["q1", "q2", "q3", "q4"],
        triage_rows=triage,
        target_pairs=10,
    )

    assert report["status"] == "complete"
    assert report["acceptance"]["rate"] == 0.5
    assert report["candidate_pool_projection"]["point_estimate_candidates"] == 20
    assert report["ai_triage"]["human_decision_contingency"] == {
        "exclude_ambiguous": {"uncertain": 1},
        "pass_grounded": {"rejected": 1},
        "repair_explanation": {"accepted": 2},
    }
    assert report["gates"]["training_gate_open"] is False
    assert "Không dùng trạng thái tạm thời" not in render_pilot_markdown(report)


def test_ai_decisions_complete_pilot_but_are_not_human_contingency() -> None:
    rows = [baseline("q1"), baseline("q2")]
    notes = [resolved(rows[0], "accepted"), resolved(rows[1], "uncertain")]
    for note in notes:
        note.update(reviewer_type="ai", reviewer="codex-visual-review")
    notes[0].update(chosen_origin_kind="synthetic", chosen_origin_source="recipe:q1",
                    chosen_recipe_sha256="a" * 64)
    report = summarize_pilot(notes, rows, ["q1", "q2"], triage_rows=[
        {"source_question_id": "q1", "ai_verdict": "repair_explanation"}])
    assert report["status"] == "complete"
    assert report["pilot"]["reviewer_types"] == {"ai": 2}
    assert report["ai_triage"]["human_decision_contingency"] == {}
    assert report["gates"]["training_gate_open"] is False
    assert "ai: 2" in render_pilot_markdown(report)


def test_invalid_source_is_reported_instead_of_estimating_yield() -> None:
    rows = [baseline("q1")]
    notes = [resolved(rows[0], "accepted")]
    notes[0]["candidate_explanation"] = "forged"

    report = summarize_pilot(notes, rows, ["q1"])

    assert report["status"] == "invalid"
    assert report["acceptance"] is None
    assert report["gates"]["next_action"] == "fix_invalid_reviews"


def test_mixed_negatives_do_not_inflate_natural_error_projection() -> None:
    rows = [baseline(f"q{i}") for i in range(4)]
    notes = [resolved(rows[0], "accepted"), resolved(rows[1], "accepted")]
    notes[1].update(
        proposed_rejected_explanation="It has scales.",
        unsupported_span="scales",
        rejected_origin_kind="synthetic",
        rejected_origin_source="controlled edit",
        rejected_recipe_sha256="c" * 64,
    )
    notes.extend([resolved(rows[2], "rejected"), resolved(rows[3], "uncertain")])
    report = summarize_pilot(notes, rows, [r["source_question_id"] for r in rows], target_pairs=10)

    assert report["status"] == "complete"
    assert report["acceptance"]["rate"] == 0.5
    assert report["natural_error_yield"]["rate"] == 0.25
    assert report["pilot"]["accepted_rejected_origins"] == {"model": 1, "synthetic": 1}
    assert report["candidate_pool_projection"]["point_estimate_candidates"] == 40
    assert "không cộng negative do người/AI tạo" in render_pilot_markdown(report)


def test_only_constructed_negatives_cannot_project_a_natural_error_pool() -> None:
    rows = [baseline("q1")]
    notes = [resolved(rows[0], "accepted")]
    notes[0].update(
        proposed_rejected_explanation="It has scales.",
        unsupported_span="scales",
        rejected_origin_kind="manual",
    )
    report = summarize_pilot(notes, rows, ["q1"])
    assert report["acceptance"]["rate"] == 1.0
    assert report["natural_error_yield"]["accepted"] == 0
    assert report["pilot"]["accepted_rejected_origins"] == {"manual": 1}
    assert report["candidate_pool_projection"] is None


def test_accepted_subset_can_be_exported_without_resolving_unused_pool() -> None:
    rows = [baseline("q1"), baseline("q2")]
    notes = [resolved(rows[0], "accepted"), rows[1]]
    report = summarize_pilot(notes, rows, ["q1"])
    assert report["gates"]["pilot_complete"] is True
    assert report["gates"]["full_review_ready_for_export"] is False
    assert report["gates"]["accepted_subset_ready_for_export"] is True
    assert report["gates"]["training_gate_open"] is False


@pytest.mark.parametrize(
    "bad_row",
    [
        {"source_question_id": "q1", "ai_verdict": "pass_grounded"},
        {"source_question_id": "q2", "ai_verdict": "made_up_verdict"},
        {"source_question_id": "outside", "ai_verdict": "pass_grounded"},
    ],
)
def test_invalid_triage_cannot_publish_a_comparison(bad_row):
    rows = [baseline("q1"), baseline("q2")]
    notes = [resolved(row, "accepted") for row in rows]
    triage = [{"source_question_id": "q1", "ai_verdict": "repair_explanation"}, bad_row]
    report = summarize_pilot(notes, rows, ["q1", "q2"], triage_rows=triage)
    assert report["status"] == "invalid"
    assert report["ai_triage"]["status"] == "invalid"
    assert report["ai_triage"]["human_decision_contingency"] == {}
    assert report["gates"]["pilot_complete"] is True
    assert report["gates"]["next_action"] == "fix_invalid_ai_triage"
    assert "Không xuất bảng đối chiếu AI" in render_pilot_markdown(report)
