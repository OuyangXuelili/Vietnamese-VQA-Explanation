import hashlib
import json
from copy import deepcopy

import pytest

from tests.unit.test_study_report import semantic_data, study_data
from vqa_explanation.evaluation import study_semantic as module
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    predictions, annotations = study_data()
    for rows in predictions.values():
        for row in rows:
            row["split"] = "validation"
    raw = tmp_path / "predictions.jsonl"
    raw.write_text("input\n")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    sources = {a: {"path": str(raw), "sha256": digest} for a in predictions}
    monkeypatch.setattr(module, "inspect_exports",
                        lambda *args: (predictions, annotations, sources))
    reports, _ = semantic_data(predictions)
    calls = []

    def evaluate(rows, **kwargs):
        calls.append(rows)
        semantic = deepcopy(reports["m0"]["explanation_bertscore"])
        protocol = semantic["protocol"]
        protocol.update(model_id=DEFAULT_BERTSCORE_MODEL,
                        model_revision=DEFAULT_BERTSCORE_REVISION,
                        num_layers=DEFAULT_BERTSCORE_LAYERS, idf=False,
                        rescale_with_baseline=False, use_fast_tokenizer=False,
                        reference_aggregation="best_match_across_all_references",
                        device=kwargs["device"], batch_size=kwargs["batch_size"])
        semantic["protocol_sha256"] = hashlib.sha256(json.dumps(
            protocol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return semantic

    return tmp_path / "semantic", calls, evaluate, raw


def test_inspect_never_calls_model_or_creates_output(prepared):
    output, calls, evaluate, _ = prepared
    result = module.run_semantic("results", "annotations", output,
                                 inspect_only=True, evaluator=evaluate)
    assert result["status"] == "inputs_verified"
    assert calls == [] and not output.exists()


def test_complete_and_resume_do_not_recompute_or_overwrite(prepared):
    output, calls, evaluate, _ = prepared
    result = module.run_semantic("results", "annotations", output, evaluator=evaluate)
    assert result["status"] == "completed" and len(calls) == 4
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    module.run_semantic("results", "annotations", output, evaluator=evaluate)
    assert len(calls) == 4
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_partial_resume_only_scores_missing_arms(prepared):
    output, calls, evaluate, _ = prepared

    def interrupted(rows, **kwargs):
        if len(calls) == 2:
            raise RuntimeError("runtime ended")
        return evaluate(rows, **kwargs)

    with pytest.raises(RuntimeError, match="runtime ended"):
        module.run_semantic("results", "annotations", output, evaluator=interrupted)
    assert (output / "m0_metrics.json").is_file()
    assert (output / "m1_metrics.json").is_file()
    assert not (output / "completed.json").exists()
    module.run_semantic("results", "annotations", output, evaluator=evaluate)
    assert len(calls) == 4


@pytest.mark.parametrize("fault", ["prediction_hash", "mean", "id", "protocol"])
def test_changed_cached_report_stops_before_more_scoring(prepared, fault):
    output, calls, evaluate, _ = prepared
    module.run_semantic("results", "annotations", output, evaluator=evaluate)
    path = output / "m0_metrics.json"
    payload = json.loads(path.read_text())
    if fault == "prediction_hash":
        payload["predictions_sha256"] = "changed"
    elif fault == "mean":
        payload["explanation_bertscore"]["mean_f1"] = .9
    elif fault == "id":
        payload["explanation_bertscore"]["items"][0]["question_id"] = "other"
    else:
        payload["explanation_bertscore"]["protocol"]["device"] = "cpu"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.run_semantic("results", "annotations", output, evaluator=evaluate)
    assert len(calls) == 4


def test_change_during_scoring_is_not_saved(prepared):
    output, _, evaluate, raw = prepared

    def changed(rows, **kwargs):
        result = evaluate(rows, **kwargs)
        raw.write_text("changed")
        return result

    with pytest.raises(ValueError, match="changed during"):
        module.run_semantic("results", "annotations", output, evaluator=changed)
    assert not (output / "m0_metrics.json").exists()


def test_different_execution_plan_refuses_existing_output(prepared):
    output, _, evaluate, _ = prepared
    module.run_semantic("results", "annotations", output, evaluator=evaluate)
    with pytest.raises(ValueError, match="another plan"):
        module.run_semantic("results", "annotations", output, batch_size=8, evaluator=evaluate)
