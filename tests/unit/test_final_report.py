"""Synthetic exports verify final-test scoring without loading any model."""

import copy
import json
from pathlib import Path
from statistics import stdev
from types import SimpleNamespace

import pytest

from scripts.report_test import paths_by_identity
from tests.unit.test_study_test import inputs
from vqa_explanation.data.io import write_json, write_jsonl
from vqa_explanation.evaluation import test_report as report
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
)
from vqa_explanation.inference.export import export_inputs, read_jsonl, select_records
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.inference.study_test import IDENTITIES, build_test_run_record
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def inference_inputs(tmp_path):
    return inputs.__wrapped__(tmp_path)


@pytest.fixture
def exports(inference_inputs, tmp_path):
    data = inference_inputs
    data.plan["metrics"] = {
        "bertscore_model": DEFAULT_BERTSCORE_MODEL,
        "bertscore_revision": DEFAULT_BERTSCORE_REVISION,
        "bertscore_layers": DEFAULT_BERTSCORE_LAYERS, "bootstrap_seed": 42,
        "bootstrap_resamples": 20}
    write_json(data.plan_path, data.plan)
    data.plan_hash = sha256_file(data.plan_path)
    selected = select_records(data.rows, "test", 0, 42)
    images = tmp_path / "images"
    hashes = {row["question_id"]: sha256_file(images / row["image_path"]) for row in selected}
    protocol = {key: value for key, value in data.plan["inference_protocol"].items()
                if key not in {"model_id", "model_revision", "seed"}}
    runs = {}
    for identity, (arm, training_seed) in IDENTITIES.items():
        directory = tmp_path / "exports" / identity
        directory.mkdir(parents=True)
        runs[identity] = directory
        args = SimpleNamespace(
            annotations=tmp_path / "data/test.jsonl", image_root=images,
            output=directory / f"{identity}.jsonl", adapter=None if identity == "m0"
            else data.adapter, split="test", limit=0, experiment_id=data.plan["runs"][identity][
                "experiment_id"], seed=42, min_pixels=200704, max_pixels=200704,
            max_new_tokens=128, load_in_4bit=True, model_id=data.plan["inference_protocol"][
                "model_id"], revision=data.plan["inference_protocol"]["model_revision"],
            export_context={"test_identity": identity, "training_seed": training_seed,
                            "plan_sha256": data.plan_hash,
                            "rubric_lock_sha256": sha256_file(data.rubric / "lock.json")})
        inputs = export_inputs(args, selected, hashes)
        manifest = {"schema_version": 1, "inputs": inputs, "protocol": protocol}
        write_json(directory / "test_run.json", build_test_run_record(args, data.plan, {
            "verified_images": 2, "checksum_manifest_sha256": "fixture"}))
        write_json(args.output.with_suffix(".jsonl.manifest.json"), manifest)
        rows = []
        wrong = (training_seed - 40 if arm == "m2" else 1 if arm == "m3" else 0)
        for i, row in enumerate(selected):
            raw = "Đáp án: " + ("xanh" if i < wrong else "đỏ") + "\nGiải thích: Vật có màu đỏ."
            parsed = parse_prediction(raw)
            rows.append({**row, **protocol, "experiment_id": args.experiment_id,
                         "model_id": args.model_id, "model_revision": args.revision,
                         "adapter": inputs["adapter"], "seed": 42,
                         "reference_answer": row["answer"],
                         "reference_explanations": row["explanations"],
                         "image_sha256": hashes[row["question_id"]], "raw_output": raw,
                         "prediction_answer": parsed.answer,
                         "prediction_explanation": parsed.explanation,
                         "format_valid": parsed.format_valid, "latency_seconds": 1.0})
        write_jsonl(args.output, rows)
        close(directory, identity)
    return data, runs


def close(directory, identity):
    path = directory / f"{identity}.jsonl"
    write_json(directory / "completion.json", {
        "identity": identity, "predictions": 1968, "predictions_sha256": sha256_file(path),
        "manifest_sha256": sha256_file(path.with_suffix(".jsonl.manifest.json")),
        "run_sha256": sha256_file(directory / "test_run.json")})


def inspect(data, runs):
    return report.inspect_test_outputs(data.plan_path, data.plan_hash,
                                       data.plan_path.parent.parent / "data/test.jsonl", runs)


def test_missing_exports_are_pending_and_do_not_produce_metrics(inference_inputs, tmp_path):
    data = inference_inputs
    runs = {name: tmp_path / "missing" / name for name in IDENTITIES}
    result = report.build_test_report(data.plan_path, data.plan_hash, tmp_path / "data/test.jsonl",
                                      runs, inspect_only=True)
    assert result["status"] == "awaiting_exports" and result["metrics_computed"] is False
    assert all(row["saved"] == 0 for row in result["progress"].values())
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match="all eight"):
        report.build_test_report(data.plan_path, data.plan_hash, tmp_path / "data/test.jsonl",
                                  runs, output_dir=output)
    assert not output.exists()


def test_complete_exports_have_one_pair_of_baselines_and_three_seed_deltas(exports):
    data, runs = exports
    plan, annotations, predictions, progress, sources = inspect(data, runs)
    assert all(row["status"] == "completed" for row in progress.values())
    result = report.analyze_test_outputs(plan, annotations, predictions, sources)
    assert result["status"] == "lexical_complete_semantic_pending"
    assert result["human_evaluation_complete"] is False
    assert result["n"] == 1968 and result["n_images"] == 2
    assert set(result["fixed_baselines"]) == {"m0", "m1"}
    deltas = result["seed_summary"]["m3_minus_m2"]["exact_match"]
    assert deltas["sample_sd"] == pytest.approx(stdev([1 / 1968, 2 / 1968, 3 / 1968]))
    assert result["bootstrap"]["m3_seed42_minus_m2_seed42"]["n_questions"] == 1968
    assert "bertscore_f1" not in result["scores"]["m3_seed42"]
    reordered = {key: list(reversed(rows)) for key, rows in predictions.items()}
    again = report.analyze_test_outputs(plan, annotations, reordered, sources)
    assert again["scores"] == result["scores"] and again["bootstrap"] == result["bootstrap"]


@pytest.mark.parametrize("fault", ["bytes", "identity", "rubric", "runtime", "references"])
def test_corrupt_or_incompatible_runs_are_not_scored(exports, fault):
    data, runs = exports
    directory = runs["m3_seed43"]
    path = directory / "m3_seed43.jsonl"
    if fault == "bytes":
        path.write_bytes(path.read_bytes() + b"changed")
    elif fault == "identity":
        record = json.loads((directory / "test_run.json").read_text("utf-8"))
        record["training_seed"] = 44
        write_json(directory / "test_run.json", record)
        close(directory, "m3_seed43")
    elif fault in {"rubric", "runtime"}:
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text("utf-8"))
        if fault == "runtime":
            manifest["inputs"]["python"] = "different"
        else:
            record = json.loads((directory / "test_run.json").read_text("utf-8"))
            record["rubric_lock_sha256"] = "e" * 64
            manifest["inputs"]["context"]["rubric_lock_sha256"] = "e" * 64
            write_json(directory / "test_run.json", record)
        write_json(path.with_suffix(".jsonl.manifest.json"), manifest)
        close(directory, "m3_seed43")
    else:
        rows = read_jsonl(path)
        rows[0]["reference_answer"] = "changed"
        write_jsonl(path, rows)
        close(directory, "m3_seed43")
    with pytest.raises(ValueError):
        inspect(data, runs)


def test_partial_export_is_never_scored_by_intersection(exports):
    data, runs = exports
    directory = runs["m2_seed44"]
    (directory / "completion.json").unlink()
    path = directory / "m2_seed44.jsonl"
    write_jsonl(path, read_jsonl(path)[:10])
    _, _, predictions, progress, _ = inspect(data, runs)
    assert len(predictions) == 7 and progress["m2_seed44"]["saved"] == 10
    with pytest.raises(ValueError, match="all eight"):
        report.build_test_report(data.plan_path, data.plan_hash,
                                  data.plan_path.parent.parent / "data/test.jsonl", runs,
                                  output_dir=directory.parent.parent / "new_report")


def test_saved_report_preserves_all_inputs_and_cannot_overwrite(exports, tmp_path):
    data, runs = exports
    before = {p: sha256_file(p) for directory in runs.values() for p in directory.iterdir()}
    output = tmp_path / "report"
    result = report.build_test_report(data.plan_path, data.plan_hash, tmp_path / "data/test.jsonl",
                                      runs, output_dir=output)
    assert result["status"] == "lexical_complete_semantic_pending"
    assert len((output / "metrics.csv").read_text("utf-8").splitlines()) == 9
    assert all(sha256_file(p) == value for p, value in before.items())
    with pytest.raises(FileExistsError):
        report.build_test_report(data.plan_path, data.plan_hash, tmp_path / "data/test.jsonl",
                                  runs, output_dir=output)


def test_semantic_reports_require_the_frozen_model_and_exact_predictions(exports):
    import hashlib

    data, runs = exports
    plan, annotations, predictions, _, sources = inspect(data, runs)
    protocol = {"model_id": DEFAULT_BERTSCORE_MODEL, "model_revision": DEFAULT_BERTSCORE_REVISION,
                "num_layers": 9, "idf": False, "rescale_with_baseline": False,
                "use_fast_tokenizer": False,
                "reference_aggregation": "best_match_across_all_references"}
    fingerprint = hashlib.sha256(json.dumps(protocol,ensure_ascii=False,sort_keys=True).encode(
        "utf-8")).hexdigest()
    semantic = {}
    for name, rows in predictions.items():
        items = [{"question_id": row["question_id"], "image_id": row["image_id"],
                  "precision": .8, "recall": .8, "f1": .8} for row in rows]
        semantic[name] = {"n": 1968, "split": "test",
                          "predictions_sha256": sources[f"{name}_predictions"]["sha256"],
                          "explanation_bertscore": {"status": "scored", "n": 1968,
                              "protocol": protocol, "protocol_sha256": fingerprint,
                              "items": items, "mean_precision": .8,
                              "mean_recall": .8, "mean_f1": .8}}
    result = report.analyze_test_outputs(plan, annotations, predictions, sources, semantic)
    assert result["status"] == "automatic_metrics_complete"
    assert result["scores"]["m0"]["bertscore_f1"] == .8
    changed = copy.deepcopy(semantic)
    changed["m3_seed44"]["predictions_sha256"] = "changed"
    with pytest.raises(ValueError, match="different predictions"):
        report.analyze_test_outputs(plan, annotations, predictions, sources, changed)
    wrong = copy.deepcopy(semantic)
    bad_protocol = {**protocol, "num_layers": 8}
    bad_hash = hashlib.sha256(json.dumps(bad_protocol,ensure_ascii=False,sort_keys=True).encode(
        "utf-8")).hexdigest()
    for row in wrong.values():
        row["explanation_bertscore"].update(protocol=bad_protocol, protocol_sha256=bad_hash)
    with pytest.raises(ValueError, match="differs from the frozen test plan"):
        report.analyze_test_outputs(plan, annotations, predictions, sources, wrong)


def test_duplicate_or_unknown_cli_mapping_is_rejected():
    assert paths_by_identity(["m0=one"]) == {"m0": Path("one")}
    for items in (["m0=one", "m0=two"], ["m3_seed45=path"], ["m0="], ["m0"]):
        with pytest.raises(ValueError):
            paths_by_identity(items)


def test_analysis_api_rejects_partial_canonical_references(exports):
    data, runs = exports
    plan, annotations, predictions, _, sources = inspect(data, runs)
    with pytest.raises(ValueError, match="complete canonical test split"):
        report.analyze_test_outputs(plan, annotations[:-1], predictions, sources)
