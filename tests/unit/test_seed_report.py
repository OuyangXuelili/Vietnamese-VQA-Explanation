import json
from copy import deepcopy
from pathlib import Path
from statistics import stdev

import pytest

from tests.unit.test_study_report import analyze_study, semantic_data, study_data
from vqa_explanation.evaluation.seed_report import render_seed_markdown, summarize_seeds
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def reports():
    predictions, annotations = study_data()
    semantic, hashes = semantic_data(predictions)
    report = analyze_study(predictions, annotations, semantic=semantic, prediction_hashes=hashes,
                           training_seed=42, resamples=2)
    report.update(n=1407, n_images=1383)
    report["sources"] = {k: {"sha256": k} for k in (
        "annotations", "canonical_all", "m1_selection", "m0", "m1")}
    report["evidence"] = {"training": {a: {"status": "passed", "global_step": 375, "epoch": 3}
                                       for a in ("m2", "m3")}}
    result = {}
    for seed, delta in ((42, .01), (43, .02), (44, .04)):
        result[seed] = deepcopy(report)
        result[seed]["training_seed"] = seed
        result[seed]["scores"]["m3"]["bertscore_f1"] = .8 + delta
    return result


def test_mean_and_sample_sd_not_pseudoreplicated_baselines(reports):
    result = summarize_seeds(reports)
    stat = result["summary"]["m3_minus_m2"]["bertscore_f1"]
    assert stat["mean"] == pytest.approx(.07 / 3)
    assert stat["sample_sd"] == pytest.approx(stdev([.01, .02, .04]))
    assert "m0" not in result["summary"]
    assert result["fixed_baselines"]["m1"] == reports[42]["scores"]["m1"]
    assert "độ lệch chuẩn mẫu" in render_seed_markdown(result)


def test_fixed_gpu_baseline_allows_roundoff_but_is_not_averaged(reports):
    original = reports[42]["scores"]["m1"]["bertscore_f1"]
    reports[44]["scores"]["m1"]["bertscore_f1"] += 3e-9
    result = summarize_seeds(reports)
    assert result["fixed_baselines"]["m1"]["bertscore_f1"] == original
    reports[44]["scores"]["m1"]["bertscore_f1"] += 2e-7
    with pytest.raises(ValueError, match="Fixed baselines"):
        summarize_seeds(reports)


@pytest.mark.parametrize("fault", ["missing_seed", "extra_seed", "wrong_seed", "test", "n",
                                  "missing_semantic", "nan", "protocol", "baseline", "data",
                                  "unfinished"])
def test_incomplete_or_changed_design_does_not_produce_summary(reports, fault):
    report = reports[44]
    if fault == "missing_seed":
        reports.pop(44)
    elif fault == "extra_seed":
        reports[45] = deepcopy(report)
    elif fault == "wrong_seed":
        report["training_seed"] = 43
    elif fault == "test":
        report["split"] = "test"
    elif fault == "n":
        report["n"] = 1406
    elif fault == "missing_semantic":
        report["scores"]["m3"].pop("bertscore_f1")
    elif fault == "nan":
        report["scores"]["m3"]["bertscore_f1"] = float("nan")
    elif fault == "protocol":
        report["semantic_protocol"]["device"] = "cpu"
    elif fault == "baseline":
        report["scores"]["m0"]["exact_match"] = .9
    elif fault == "data":
        report["sources"]["annotations"]["sha256"] = "other"
    else:
        report["evidence"]["training"]["m3"]["global_step"] = 374
    with pytest.raises(ValueError):
        summarize_seeds(reports)


def test_file_report_passes_path_objects_to_evidence_verifier(tmp_path, monkeypatch):
    from vqa_explanation.evaluation import seed_report as module

    inputs = {}
    for name in ("annotations", "canonical_all", "study_config", "m0", "m1", "m2", "m3"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text('{}\n', encoding="utf-8")
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"sources": inputs}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "report.json": sha256_file(report_path)}}), encoding="utf-8")

    def verify(folder, annotations, predictions, study, canonical):
        assert isinstance(study, Path) and isinstance(canonical, Path)
        assert canonical.is_file() and study.is_file()
        raise RuntimeError("Reached original-evidence verification")

    monkeypatch.setattr(module, "verify_evidence", verify)
    output = tmp_path / "summary"
    with pytest.raises(RuntimeError, match="original-evidence"):
        module.build_seed_report({42: report_path}, output)
    assert not output.exists()
