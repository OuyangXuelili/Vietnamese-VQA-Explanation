import json

import pytest

from tests.unit.test_blind_review import blind_bundle as blind_bundle  # noqa: F401
from tests.unit.test_blind_review import ratings
from vqa_explanation.data.io import read_jsonl, write_jsonl
from vqa_explanation.evaluation.blind_review import summarize_blind_review
from vqa_explanation.evaluation.review_progress import (
    inspect_review_progress,
    refresh_review_pages,
    save_review_progress,
)
from vqa_explanation.utils.run_provenance import sha256_file


def test_blank_test_forms_report_pending_without_quality_scores(blind_bundle, tmp_path):
    report = inspect_review_progress(blind_bundle)
    assert report["split"] == "test" and report["status"] == "awaiting_ratings"
    assert report["summary"] is None and report["agreement"] is None
    assert all(row["complete"] == row["images_complete"] == 0
               and row["remaining"] == 12 for row in report["raters"].values())
    output = save_review_progress(report, tmp_path / "progress")
    assert json.loads((output / "report.json").read_text("utf-8")) == report
    with pytest.raises(FileExistsError):
        save_review_progress(report, output)


def test_partial_ratings_count_images_and_are_not_scored(blind_bundle):
    paths = ratings(blind_bundle)
    rows = list(read_jsonl(paths["rater2"]))
    rows[0]["image_evidence"] = ""
    write_jsonl(paths["rater2"], rows)
    report = inspect_review_progress(blind_bundle, paths)
    assert report["raters"]["rater1"]["images_complete"] == 3
    assert report["raters"]["rater2"]["complete"] == 11
    assert report["raters"]["rater2"]["images_complete"] == 2
    assert report["summary"] is None and report["agreement"] is None
    with pytest.raises(ValueError, match="pending or incomplete"):
        summarize_blind_review(blind_bundle, paths)


def test_ready_status_does_not_invent_independence_or_agreement(blind_bundle):
    files = ratings(blind_bundle)
    report = inspect_review_progress(blind_bundle, {"rater1": files["rater1"]})
    assert report["status"] == "ready_to_score"
    assert report["summary"] is None and report["agreement"] is None


@pytest.mark.parametrize("fault", ["missing", "duplicate", "output", "same_reviewer"])
def test_progress_rejects_bad_coverage_source_or_reviewer(blind_bundle, fault):
    paths = ratings(blind_bundle)
    rows = list(read_jsonl(paths["rater2"]))
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "output":
        rows[0]["prediction_answer"] = "changed"
    else:
        for row in rows:
            row["reviewer"] = "rater1"
    write_jsonl(paths["rater2"], rows)
    with pytest.raises(ValueError):
        inspect_review_progress(blind_bundle, paths)


def test_stale_progress_and_writes_inside_the_bundle_are_rejected(blind_bundle, tmp_path):
    report = inspect_review_progress(blind_bundle)
    with pytest.raises(ValueError, match="overlaps"):
        save_review_progress(report, blind_bundle / "progress")
    (blind_bundle / "rater1/index.html").write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="changed after"):
        save_review_progress(report, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()


def test_refresh_keeps_blinding_content_and_scores_while_preserving_old_bundle(
    blind_bundle, tmp_path,
):
    before = {p: sha256_file(p) for p in blind_bundle.rglob("*") if p.is_file()}
    old_scores = summarize_blind_review(blind_bundle, ratings(blind_bundle))
    new = refresh_review_pages(blind_bundle, tmp_path / "refreshed")
    manifest = json.loads((new / "private/manifest.json").read_text("utf-8"))
    assert manifest["origin_manifest_sha256"] == before[blind_bundle / "private/manifest.json"]
    for rater in ("rater1", "rater2"):
        assert (new / rater / "blank_ratings.jsonl").read_bytes() == (
            blind_bundle / rater / "blank_ratings.jsonl").read_bytes()
    assert (new / "private/model_key.jsonl").read_bytes() == (
        blind_bundle / "private/model_key.jsonl").read_bytes()
    assert summarize_blind_review(new, ratings(new)) == old_scores
    assert all(sha256_file(p) == sha for p, sha in before.items())
    with pytest.raises(FileExistsError):
        refresh_review_pages(blind_bundle, new)
