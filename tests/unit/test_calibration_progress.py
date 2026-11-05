import json

import pytest

from tests.unit.test_rubric_calibration import filled_ratings
from tests.unit.test_rubric_calibration import practice as practice  # noqa: F401
from vqa_explanation.data.io import read_jsonl, write_jsonl
from vqa_explanation.evaluation.calibration_progress import (
    inspect_calibration,
    save_calibration_progress,
)
from vqa_explanation.utils.run_provenance import sha256_file


def change(path, update):
    rows = list(read_jsonl(path))
    update(rows)
    write_jsonl(path, rows)


def test_blank_drafts_are_not_scored_or_frozen(practice, tmp_path):
    before = {p: sha256_file(p) for p in practice.rglob("*") if p.is_file()}
    result = inspect_calibration(practice)
    assert result["status"] == "awaiting_ratings"
    assert result["summary"] is None and result["disagreements"] is None
    assert result["rubric_frozen"] is False and result["test_predictions_generated"] is False
    assert all(r["complete"] == 0 and r["remaining"] == 120 for r in result["raters"].values())
    output = save_calibration_progress(result, tmp_path / "report")
    assert not (output / "disagreements.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))["files"]
    assert all(sha256_file(output / name) == expected for name, expected in manifest.items())
    assert all(sha256_file(path) == expected for path, expected in before.items())
    with pytest.raises(FileExistsError):
        save_calibration_progress(result, output)


def test_partial_draft_lists_missing_fields_without_revealing_other_ratings(practice):
    paths = filled_ratings(practice)
    change(paths["rater2"], lambda rows: rows[0].update(image_evidence=""))
    result = inspect_calibration(practice, paths)
    assert result["status"] == "awaiting_ratings"
    assert result["raters"]["rater1"]["complete"] == 120
    assert result["raters"]["rater2"]["complete"] == 119
    assert result["raters"]["rater2"]["pending_entries"][0]["missing"] == ["image_evidence"]
    assert result["summary"] is None and result["disagreements"] is None


def test_matching_complete_ratings_are_ready_for_discussion_not_locked(practice, tmp_path):
    result = inspect_calibration(practice, filled_ratings(practice))
    assert result["status"] == "ready_for_discussion"
    assert result["disagreements"] == []
    assert result["summary"]["status"] == "calibration_complete"
    assert result["rubric_frozen"] is False
    output = save_calibration_progress(result, tmp_path / "complete")
    assert (output / "disagreements.csv").is_file()
    assert not (output / "lock.json").exists()


def test_disagreements_align_different_blind_orders_by_the_private_key(practice):
    paths = filled_ratings(practice)
    key = list(read_jsonl(practice / "bundle/private/model_key.jsonl"))
    chosen = next(r for r in key if r["rater_id"] == "rater2")
    change(paths["rater2"], lambda rows: next(
        row for row in rows if row["entry_id"] == chosen["entry_id"]
    ).update(explanation_support=0, unsupported_detail="yes"))
    result = inspect_calibration(practice, paths)
    assert len(result["disagreements"]) == 2
    assert {r["field"] for r in result["disagreements"]} == {
        "explanation_support", "unsupported_detail"}
    assert all(r["question_id"] == chosen["question_id"]
               and r["experiment_id"] == chosen["experiment_id"] for r in result["disagreements"])


def test_one_rater_has_no_fabricated_agreement(practice):
    result = inspect_calibration(practice, {"rater1": filled_ratings(practice)["rater1"]})
    assert result["status"] == "one_rater_complete"
    assert result["summary"]["agreement"] is None
    assert result["disagreements"] is None


@pytest.mark.parametrize("field,value", [("explanation_support", True),
                                        ("explanation_support", 2.0),
                                        ("answer_assessment", "maybe"),
                                        ("reviewed_at", "2026-10-27T12:00:00")])
def test_invalid_nonempty_labels_are_rejected(practice, field, value):
    paths = filled_ratings(practice)
    change(paths["rater1"], lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError):
        inspect_calibration(practice, paths)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "output", "reviewer", "shared"])
def test_bad_coverage_mutated_outputs_and_reviewer_identity_are_rejected(practice, fault):
    paths = filled_ratings(practice)
    def mutate(rows):
        if fault == "missing":
            rows.pop()
        elif fault == "duplicate":
            rows[0] = rows[1].copy()
        elif fault == "output":
            rows[0]["prediction_answer"] = "changed"
        elif fault == "reviewer":
            rows[0]["reviewer"] = "another-person"
        else:
            for row in rows:
                row["reviewer"] = "rater2"
    change(paths["rater1"], mutate)
    with pytest.raises(ValueError):
        inspect_calibration(practice, paths)


def test_modified_bundle_or_changed_ratings_cannot_be_saved_as_a_current_report(practice, tmp_path):
    paths = filled_ratings(practice)
    result = inspect_calibration(practice, paths)
    change(paths["rater1"], lambda rows: rows[0].update(image_evidence="new note"))
    with pytest.raises(ValueError, match="changed after"):
        save_calibration_progress(result, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()
    (practice / "bundle/rater1/index.html").write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="bundle changed"):
        inspect_calibration(practice)


def test_report_cannot_be_written_into_the_historical_calibration(practice):
    result = inspect_calibration(practice)
    with pytest.raises(ValueError, match="overlaps"):
        save_calibration_progress(result, practice / "new_report")


def test_rating_change_between_shared_inspection_and_summary_is_rejected(practice, monkeypatch):
    import vqa_explanation.evaluation.calibration_progress as module

    paths = filled_ratings(practice)
    original = module.inspect_review_progress

    def inspect_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        change(paths["rater1"], lambda rows: rows[0].update(image_evidence="changed during read"))
        return result

    monkeypatch.setattr(module, "inspect_review_progress", inspect_then_change)
    with pytest.raises(ValueError, match="changed during"):
        inspect_calibration(practice, paths)
