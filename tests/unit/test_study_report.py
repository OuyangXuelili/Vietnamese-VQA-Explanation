import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.unit.test_validation_report import data
from vqa_explanation.evaluation.study_report import (
    analyze_study,
    render_study_cases,
    render_study_markdown,
    verify_evidence,
)
from vqa_explanation.inference.export import select_records
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.inference.study_validation import (
    MODEL_ID,
    MODEL_REVISION,
    main_study_settings,
)
from vqa_explanation.utils.run_provenance import sha256_file


def study_data():
    m0, m1, annotations = data()
    for rows in (m0, m1):
        for row in rows:
            row["latency_seconds"] = 1.2
    predictions = {"m0": m0, "m1": m1, "m2": deepcopy(m0), "m3": deepcopy(m1)}
    for arm, experiment in (("m2", "M2_SFT_CHOSEN"), ("m3", "M3_DPO")):
        for row in predictions[arm]:
            row.update(experiment_id=experiment, adapter=f"{arm}/checkpoint-375")
    return predictions, annotations


def semantic_data(predictions):
    hashes = {arm: hashlib.sha256(arm.encode()).hexdigest() for arm in predictions}
    protocol = {"metric": "bertscore", "model_id": "fixture", "device": "cuda"}
    digest = hashlib.sha256(json.dumps(protocol, ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()
    reports = {}
    for arm, rows in predictions.items():
        items = [{"question_id": r["question_id"], "image_id": r["image_id"],
                  "precision": .8, "recall": .8, "f1": .8} for r in rows]
        reports[arm] = {"n": 4, "split": "validation", "predictions_sha256": hashes[arm],
                        "explanation_bertscore": {
                            "status": "scored", "n": 4, "items": items, "protocol": protocol,
                            "protocol_sha256": digest, "mean_precision": .8,
                            "mean_recall": .8, "mean_f1": .8}}
    return reports, hashes


def test_four_arm_analysis_keeps_all_groups_and_fixed_examples():
    predictions, annotations = study_data()
    report = analyze_study(predictions, annotations, resamples=30)
    assert set(report["scores"]) == {"m0", "m1", "m2", "m3"}
    assert len(report["bootstrap"]) == 6
    assert report["em_transitions_m2_m3"] == dict(both_match=1, gain=1, loss=1, neither_match=1)
    assert report["bootstrap"]["m3_minus_m2"]["metrics"]["exact_match"][
        "delta_right_minus_left"] == 0
    reordered = analyze_study({a: r[::-1] for a, r in predictions.items()}, annotations[::-1],
                              resamples=30)
    assert report["case_selection"] == reordered["case_selection"]
    assert report["bootstrap"] == reordered["bootstrap"]


@pytest.mark.parametrize("fault", ["missing_arm", "missing_row", "duplicate", "test",
                                  "reference", "protocol", "format", "wrong_arm"])
def test_analysis_rejects_incompatible_inputs(fault):
    predictions, annotations = study_data()
    if fault == "missing_arm":
        predictions.pop("m2")
    elif fault == "missing_row":
        predictions["m3"].pop()
    elif fault == "duplicate":
        predictions["m3"].append(predictions["m3"][0])
    elif fault == "test":
        annotations[0]["split"] = "test"
    elif fault == "reference":
        predictions["m3"][0]["reference_answer"] = "changed"
    elif fault == "protocol":
        for row in predictions["m3"]:
            row["seed"] = 43
    elif fault == "format":
        predictions["m3"][0]["format_valid"] = "true"
    else:
        for row in predictions["m3"]:
            row["experiment_id"] = "M2_SFT_CHOSEN"
    with pytest.raises(ValueError):
        analyze_study(predictions, annotations, resamples=20)


def test_semantic_reports_are_bound_to_exact_predictions_and_all_ids():
    predictions, annotations = study_data()
    reports, hashes = semantic_data(predictions)
    result = analyze_study(predictions, annotations, semantic=reports,
                           prediction_hashes=hashes, resamples=20)
    assert result["scores"]["m3"]["bertscore_f1"] == .8
    assert result["bootstrap"]["m3_minus_m2"]["metrics"]["bertscore_f1"]["ci95"] == [0, 0]
    reports["m3"]["predictions_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different predictions"):
        analyze_study(predictions, annotations, semantic=reports, prediction_hashes=hashes)


@pytest.mark.parametrize("fault", ["missing_arm", "id", "protocol", "aggregate"])
def test_rejects_partial_or_mismatched_semantic_reports(fault):
    predictions, annotations = study_data()
    reports, hashes = semantic_data(predictions)
    if fault == "missing_arm":
        reports.pop("m3")
    elif fault == "id":
        reports["m3"]["explanation_bertscore"]["items"][0]["image_id"] = "other"
    elif fault == "protocol":
        reports["m3"]["explanation_bertscore"]["protocol"]["device"] = "cpu"
    else:
        reports["m3"]["explanation_bertscore"]["mean_f1"] = .9
    with pytest.raises(ValueError):
        analyze_study(predictions, annotations, semantic=reports, prediction_hashes=hashes)


def test_report_labels_unknown_semantic_scores_and_escapes_html():
    predictions, annotations = study_data()
    predictions["m3"][0]["prediction_explanation"] = "<script>alert(1)</script>"
    result = analyze_study(predictions, annotations, resamples=20)
    text = render_study_markdown(result)
    assert "Chưa đo" in text and "chưa phải kết quả test" in text
    html = render_study_cases(
        result, {a: {r["question_id"]: r for r in rows} for a, rows in predictions.items()},
        {r["question_id"]: r for r in annotations})
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "M2 · EM" in html and "M3 · EM" in html


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    from vqa_explanation.evaluation import study_report as module

    predictions, annotations = study_data()
    all_path = tmp_path / "all.jsonl"
    all_path.write_text("canonical", encoding="utf-8")
    selection = tmp_path / "m1_selection.json"
    selection.write_text("selection", encoding="utf-8")
    study = "configs/experiments/preference_study_v2.yaml"
    settings = main_study_settings(study)
    monkeypatch.setattr(module, "VALIDATION_SHA256", "validation_fixture")
    folder = tmp_path / "validation"
    folder.mkdir()
    expected_hash = hashlib.sha256(json.dumps(
        select_records(annotations, "validation", 0, 42), ensure_ascii=False,
        sort_keys=True).encode()).hexdigest()
    parent = {"files": {"adapter_config.json": "config", "adapter_model.safetensors": "m1"}}
    for arm, rows in predictions.items():
        for row in rows:
            row["generation_config"]["num_beams"] = 1
            row["raw_output"] = "Đáp án: mèo\nGiải thích: Có mèo."
            parsed = parse_prediction(row["raw_output"])
            row.update(prediction_answer=parsed.answer, prediction_explanation=parsed.explanation,
                       format_valid=parsed.format_valid)
        protocol = {k: rows[0][k] for k in ("prompt_sha256", "compute_dtype",
                                          "generation_config", "max_pixels", "min_pixels")}
        protocol["generation_config"]["num_beams"] = 1
        inputs = {k: rows[0][k] for k in ("model_id", "model_revision", "experiment_id",
                                         "adapter", "seed", "max_pixels", "min_pixels",
                                         "max_new_tokens", "load_in_4bit")}
        inputs.update(model_id=MODEL_ID, model_revision=MODEL_REVISION, split="validation",
                      limit=0, adapter_files=None if arm == "m0" else parent["files"],
                      annotations_sha256="validation_fixture", selected_sha256=expected_hash,
                      image_hashes={r["question_id"]: r["image_sha256"] for r in rows},
                      source_files={"pyproject.toml": sha256_file(Path("pyproject.toml"))},
                      python="3.13", versions={"torch": "fixture"})
        for row in rows:
            row.update(model_id=MODEL_ID, model_revision=MODEL_REVISION)
        manifest = {"schema_version": 1, "inputs": inputs, "protocol": protocol}
        (folder / f"{arm}.jsonl.manifest.json").write_text(json.dumps(manifest), "utf-8")
        if arm not in ("m2", "m3"):
            continue
        stage = tmp_path / "training" / arm
        (stage / "training").mkdir(parents=True)
        plan = {"pairs": 1000, "config": settings[arm]["config"],
                "comparison_protocol": settings[arm]["protocol"],
                "expected_updates": 375, "annotations_sha256": sha256_file(all_path),
                "preferences_manifest_sha256": "frozen", "preferences_files": {"pairs": "sha"},
                "m1_adapter": parent, "m1_selection_sha256": sha256_file(selection),
                "source_files": {"trainer.py": "sha"},
                "visual_quality": {"sha256": settings[arm]["quality_sha256"], "ready": True,
                                   "confirmed": 1000, "unique_images": 1000,
                                   "required_review_version": 2},
                "model": {"id": MODEL_ID, "revision": MODEL_REVISION}}
        plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        result = {"status": "passed", "global_step": 375, "epoch": 3, "training_loss": 1,
                  "policy_updated": True, "saved_policy_verified": True,
                  "reference_unchanged": True, "plan_sha256": plan["sha256"],
                  "output_adapter": parent}
        for name, payload in (("data_plan.json", plan), ("training_result.json", result),
                              ("training/trainer_state.json", {
                                  "global_step": 375, "max_steps": 375, "epoch": 3})):
            (stage / name).write_text(json.dumps(payload), "utf-8")
    return tmp_path, annotations, predictions, study, all_path


def test_training_and_export_evidence_is_checked_without_loading_weights(evidence):
    result = verify_evidence(*evidence)
    assert result["training"]["m3"]["global_step"] == 375


@pytest.mark.parametrize("fault", ["state", "weights", "model", "versions", "source",
                                  "canonical", "selection", "raw_output", "sampling"])
def test_changed_evidence_blocks_report(evidence, fault):
    folder, _, predictions, _, canonical = evidence
    path = folder / "validation/m3.jsonl.manifest.json"
    value = json.loads(path.read_text())
    if fault == "state":
        path = folder / "training/m3/training/trainer_state.json"
        value = json.loads(path.read_text())
        value["global_step"] = 374
    elif fault == "weights":
        value["inputs"]["adapter_files"]["adapter_model.safetensors"] = "wrong"
    elif fault == "model":
        value["inputs"]["model_id"] = "other"
    elif fault == "versions":
        value["inputs"]["versions"]["torch"] = "other"
    elif fault == "source":
        value["inputs"]["source_files"]["pyproject.toml"] = "wrong"
    elif fault == "canonical":
        canonical.write_text("changed", "utf-8")
    elif fault == "selection":
        (folder / "m1_selection.json").write_text("changed", "utf-8")
    elif fault == "raw_output":
        predictions["m3"][0]["prediction_answer"] = "dog"
    else:
        value["protocol"]["generation_config"]["do_sample"] = True
    path.write_text(json.dumps(value), "utf-8")
    with pytest.raises(ValueError):
        verify_evidence(*evidence)
