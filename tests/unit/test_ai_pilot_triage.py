import json
from pathlib import Path

import pytest

from scripts.build_ai_pilot_triage import (
    build_ai_triage,
    load_visual_review,
    validate_source_hashes,
)
from vqa_explanation.utils.run_provenance import sha256_file


def test_triage_requires_explicit_visual_decisions(tmp_path):
    with pytest.raises(TypeError, match="visual_review_path"):
        build_ai_triage(tmp_path, tmp_path / "pilot.json", tmp_path / "out")


def test_source_hash_validation_rejects_changed_file(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": 1}\n', encoding="utf-8")
    pilot = {"source_inputs_sha256": sha256_file(source)}
    paths = {"source_inputs_sha256": source}

    assert validate_source_hashes(pilot, paths) == pilot

    source.write_text('{"id": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source_inputs_sha256"):
        validate_source_hashes(pilot, paths)


@pytest.fixture
def visual_review_input(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_ai_pilot_triage.project_root", lambda: tmp_path)
    source = tmp_path / "triage.jsonl"
    source.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    review = {
        "schema": "ai_visual_pilot_review_v1", "status": "pending_user_review",
        "annotator_type": "ai", "reviewer": None, "human_verified": False,
        "training_eligible": False, "source_triage": "triage.jsonl",
        "source_triage_sha256": sha256_file(source), "count": 1,
        "verdict_counts": {"repair_explanation": 1},
        "records": [{
            "question_id": "q1", "ai_verdict": "repair_explanation",
            "answer_assessment_ai": "correct", "image_evidence_ai": "Visible shower.",
            "reviewer": None, "human_verified": False, "training_eligible": False,
            "proposed_chosen_explanation": "shower", "unsupported_span": "bathtub",
            "error_type": "object",
        }],
    }
    return tmp_path / "review.json", review, {"q1": {"prediction_explanation": "bathtub"}}


def test_visual_review_retains_unnamed_pending_provenance(visual_review_input):
    path, review, candidates = visual_review_input
    path.write_text(json.dumps(review), encoding="utf-8")
    result = load_visual_review(path, ["q1"], candidates)
    assert result["q1"]["reviewer"] is None
    assert result["q1"]["human_verified"] is False
    assert result["q1"]["training_eligible"] is False
    with pytest.raises(ValueError, match="frozen pilot IDs"):
        load_visual_review(path, ["other"], candidates)
    (path.parent / "triage.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_visual_review(path, ["q1"], candidates)


@pytest.mark.parametrize("issue", ["answer_support", "contradiction", "unsupported_detail"])
def test_visual_review_preserves_distinct_explanation_issues(visual_review_input, issue):
    path, review, candidates = visual_review_input
    review["records"][0]["explanation_issue_ai"] = issue
    path.write_text(json.dumps(review), encoding="utf-8")
    assert load_visual_review(path, ["q1"], candidates)["q1"]["explanation_issue_ai"] == issue
    review["records"][0]["explanation_issue_ai"] = "all_weak_text_is_hallucination"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="explanation issue"):
        load_visual_review(path, ["q1"], candidates)


@pytest.mark.parametrize("field,value", [
    ("reviewer", "R01"), ("human_verified", True), ("training_eligible", True),
    ("unsupported_span", "not present"), ("proposed_chosen_explanation", "bathtub"),
    ("answer_assessment_ai", "uncertain"), ("image_evidence_ai", ""),
])
def test_visual_review_rejects_false_confirmation_and_invalid_repairs(
    visual_review_input, field, value,
):
    path, review, candidates = visual_review_input
    review["records"][0][field] = value
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError):
        load_visual_review(path, ["q1"], candidates)
