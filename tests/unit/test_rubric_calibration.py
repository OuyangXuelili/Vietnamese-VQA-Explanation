import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation import rubric_calibration as calibration
from vqa_explanation.evaluation.blind_review import freeze_sample, prepare_blind_review
from vqa_explanation.inference.study_validation import ARMS
from vqa_explanation.utils.run_provenance import sha256_file

RUBRIC = Path(__file__).resolve().parents[2] / "configs/evaluation/human_rubric_v1.yaml"


@pytest.fixture
def practice(tmp_path):
    rows, paths = [], {}
    for i in range(30):
        image = tmp_path / "images" / f"{i}.jpg"
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), color=(i * 7, 10, 20)).save(image)
        rows.append({"question_id": f"q{i}", "image_id": str(i),
                     "image_path": f"images/{i}.jpg", "question": "Animal?", "answer": "cat",
                     "explanations": ["Fur."], "split": "validation"})
    write_jsonl(tmp_path / "all.jsonl", rows)
    for arm, role in ARMS.items():
        predictions = [{**{k: r[k] for k in (
            "question_id", "image_id", "image_path", "question", "split")},
            "reference_answer": r["answer"], "reference_explanations": r["explanations"],
            "prediction_answer": "cat", "prediction_explanation": "Fur.",
            "image_sha256": sha256_file(tmp_path / r["image_path"]), "experiment_id": role,
            "adapter": None if arm == "m0" else arm, "model_id": "fixture",
            "model_revision": "fixture", "seed": 42, "max_pixels": 200704,
            "min_pixels": 200704, "max_new_tokens": 128, "prompt_sha256": "a" * 64,
            "generation_config": {"do_sample": False}, "load_in_4bit": True,
            "compute_dtype": "float16"} for r in rows]
        paths[role] = tmp_path / f"{arm}.jsonl"
        write_jsonl(paths[role], predictions)
    output = tmp_path / "practice"
    freeze_sample(tmp_path / "all.jsonl", output / "sample.json", count=30, split="validation")
    prepare_blind_review(tmp_path / "all.jsonl", output / "sample.json", paths, tmp_path,
                         output / "bundle", expected_split="validation")
    manifest = {"schema": "rubric_calibration_v1", "split": "validation", "images": 30,
                "outputs_per_rater": 120, "files": {
                    p.relative_to(output).as_posix(): sha256_file(p)
                    for p in output.rglob("*") if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return output


def filled_ratings(practice):
    paths = {}
    for rater in ("rater1", "rater2"):
        rows = list(read_jsonl(practice / "bundle" / rater / "blank_ratings.jsonl"))
        for row in rows:
            row.update(review_status="reviewed", reviewer=rater, explanation_support=2,
                       answer_assessment="correct", unsupported_detail="no",
                       reviewed_at="2026-10-23T12:00:00+07:00", image_evidence="Fixture fur.")
        paths[rater] = practice.parent / f"{rater}.jsonl"
        write_jsonl(paths[rater], rows)
    return paths


def test_rubric_accepts_declared_labels_and_keeps_uncertain():
    rubric = calibration.load_rubric(RUBRIC)
    assert set(rubric["explanation_support"]) == {0, 1, 2}
    assert "uncertain" in rubric["answer_assessment"]
    assert "uncertain" in rubric["unsupported_detail"]


@pytest.mark.parametrize("field,value", [("primary_metric", "length"), ("test_images", 30),
                                        ("answer_assessment", {"correct": "ok"}),
                                        ("instructions", [])])
def test_changed_design_or_missing_labels_are_rejected(tmp_path, field, value):
    rubric = yaml.safe_load(RUBRIC.read_text("utf-8"))
    rubric[field] = value
    path = tmp_path / "rubric.yaml"
    path.write_text(yaml.safe_dump(rubric), "utf-8")
    with pytest.raises(ValueError):
        calibration.load_rubric(path)


def test_pending_labels_cannot_freeze_a_rubric(practice):
    paths = {rater: practice / "bundle" / rater / "blank_ratings.jsonl"
             for rater in ("rater1", "rater2")}
    with pytest.raises(ValueError, match="pending or incomplete"):
        calibration.freeze_rubric(practice, paths, RUBRIC, practice.parent / "locked")
    assert not (practice.parent / "locked").exists()


def test_complete_practice_freezes_a_new_rubric_without_altering_inputs(practice):
    paths = filled_ratings(practice)
    before = {path: sha256_file(path) for path in [*paths.values(), RUBRIC]}
    output = practice.parent / "locked"
    result = calibration.freeze_rubric(practice, paths, RUBRIC, output)
    assert result["status"] == "frozen" and result["raters"] == 2
    assert result["rubric_sha256"] == sha256_file(output / "rubric.yaml")
    summary = json.loads((output / "calibration_summary.json").read_text("utf-8"))
    assert summary["split"] == "validation" and summary["status"] == "calibration_complete"
    assert all(sha256_file(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        calibration.freeze_rubric(practice, paths, RUBRIC, output)


def test_one_rater_is_explicit_and_has_no_fake_agreement(practice):
    paths = filled_ratings(practice)
    output = practice.parent / "one_rater"
    result = calibration.freeze_rubric(practice, {"rater1": paths["rater1"]}, RUBRIC, output)
    assert result["raters"] == 1
    summary = json.loads((output / "calibration_summary.json").read_text("utf-8"))
    assert summary["agreement"] is None


def test_modified_calibration_source_is_rejected(practice):
    paths = filled_ratings(practice)
    (practice / "bundle/rater1/index.html").write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="bundle changed"):
        calibration.freeze_rubric(practice, paths, RUBRIC, practice.parent / "locked")


def test_output_must_not_overlap_inputs_or_source(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for path in (root, root / "src/new", root / "data/new", tmp_path):
        with pytest.raises((ValueError, FileExistsError)):
            calibration.check_new_output(path, (tmp_path / "input.json",))


def test_wrong_representative_seed_stops_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "verified_validation", lambda *args: (
        {}, {}, {"training_seed": 43, "training_learning_rate": 1e-5}))
    with pytest.raises(ValueError, match="precommitted main seed 42"):
        calibration.prepare_calibration(tmp_path / "evidence", tmp_path / "val.jsonl",
                                        tmp_path / "all.jsonl", tmp_path / "images",
                                        tmp_path / "study.yaml", RUBRIC, tmp_path / "output")
    assert not (tmp_path / "output").exists()
