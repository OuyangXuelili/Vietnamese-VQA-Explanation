"""Score a complete validation export without loading a VQA model."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

from filelock import FileLock

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.evaluation.comparison import compare_predictions
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
    compare_bertscore_reports,
    evaluate_bertscore,
)
from vqa_explanation.inference.export import select_records, validate_saved_predictions
from vqa_explanation.inference.study_validation import ARMS, VALIDATION_SHA256
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def inspect_exports(results_dir, annotations_path):
    results = Path(results_dir)
    annotations_path = Path(annotations_path)
    if sha256_file(annotations_path) != VALIDATION_SHA256:
        raise ValueError("Require the frozen full validation annotations")
    annotations = list(read_jsonl(annotations_path))
    if (len(annotations) != 1407 or len({r["question_id"] for r in annotations}) != 1407
            or any(r["split"] != "validation" for r in annotations)):
        raise ValueError("Require 1,407 unique validation questions")
    selected = select_records(annotations, "validation", 0, 42)
    selected_hash = hashlib.sha256(json.dumps(
        selected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    predictions, sources, manifests = {}, {}, {}
    sources["annotations"] = {"path": str(annotations_path.resolve()),
                              "sha256": sha256_file(annotations_path)}
    for arm in ARMS:
        path = results / f"{arm}.jsonl"
        manifest_path = path.with_suffix(".jsonl.manifest.json")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        inputs = manifest["inputs"]
        if (manifest.get("schema_version") != 1
                or inputs.get("annotations_sha256") != VALIDATION_SHA256
                or inputs.get("selected_sha256") != selected_hash
                or inputs.get("experiment_id") != ARMS[arm]
                or inputs.get("split") != "validation" or inputs.get("limit") != 0
                or inputs.get("seed") != 42):
            raise ValueError(f"{arm}: export manifest differs from the validation protocol")
        rows = list(read_jsonl(path))
        if len(validate_saved_predictions(rows, selected, manifest)) != 1407:
            raise ValueError(f"{arm}: incomplete validation export")
        predictions[arm], manifests[arm] = rows, manifest
        for key, source in ((arm, path), (f"{arm}_manifest", manifest_path)):
            sources[key] = {"path": str(source.resolve()), "sha256": sha256_file(source)}
    for left, right in combinations(ARMS, 2):
        comparison = compare_predictions(
            predictions[left], predictions[right],
            expected_annotations={r["question_id"]: r for r in annotations},
            expected_experiments=(ARMS[left], ARMS[right]))
        if not comparison["comparable"]:
            raise ValueError(f"{left}/{right}: incompatible validation exports")
    for arm in ARMS:
        if any(manifests[arm]["inputs"][key] != manifests["m0"]["inputs"][key]
               for key in ("source_files", "python", "versions")):
            raise ValueError(f"{arm}: inference source or runtime differs")
    return predictions, annotations, sources


def verify_semantic(report, rows, prediction_hash, *, device, batch_size):
    result, _, _ = compare_bertscore_reports(
        report, report, left_predictions_sha256=prediction_hash,
        right_predictions_sha256=prediction_hash, expected_split="validation",
        expected_question_images={r["question_id"]: r["image_id"] for r in rows})
    expected = {
        "model_id": DEFAULT_BERTSCORE_MODEL, "model_revision": DEFAULT_BERTSCORE_REVISION,
        "num_layers": DEFAULT_BERTSCORE_LAYERS, "idf": False,
        "rescale_with_baseline": False, "use_fast_tokenizer": False,
        "reference_aggregation": "best_match_across_all_references",
        "device": device, "batch_size": batch_size,
    }
    if any(result["protocol"].get(key) != value for key, value in expected.items()):
        raise ValueError("Saved semantic report uses a different protocol")
    return result["protocol"]


def run_semantic(results_dir, annotations_path, output_dir, *, device="cuda", batch_size=16,
                 cache_dir=None, inspect_only=False, evaluator=evaluate_bertscore):
    predictions, _, sources = inspect_exports(results_dir, annotations_path)
    if device not in {"cuda", "cpu"} or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Require an explicit device and positive batch size")
    output = Path(output_dir)
    plan = {"schema": "study_semantic_plan_v1", "sources": sources,
            "device": device, "batch_size": batch_size,
            "model_id": DEFAULT_BERTSCORE_MODEL, "model_revision": DEFAULT_BERTSCORE_REVISION,
            "num_layers": DEFAULT_BERTSCORE_LAYERS,
            "evaluator_source": {name: sha256_file(project_root() / name) for name in (
                "src/vqa_explanation/evaluation/study_semantic.py",
                "src/vqa_explanation/evaluation/semantic_metrics.py",
                "requirements-cloud-semantic.txt")}}
    if inspect_only:
        return {"status": "inputs_verified", "questions_per_arm": 1407, "plan": plan}
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        plan_path = output / "plan.json"
        if output.exists():
            if not plan_path.is_file() or json.loads(plan_path.read_text("utf-8")) != plan:
                raise ValueError("Semantic output belongs to another plan; use a new directory")
        else:
            output.mkdir()
            with plan_path.open("x", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
        reports = {}
        for arm, rows in predictions.items():
            path = output / f"{arm}_metrics.json"
            if path.exists():
                report = json.loads(path.read_text("utf-8"))
                print(f"Verify saved BERTScore {arm}", flush=True)
            else:
                print(f"BERTScore {arm}: {len(rows)} validation questions", flush=True)
                report = {"n": len(rows), "split": "validation",
                          "predictions_sha256": sources[arm]["sha256"],
                          "explanation_bertscore": evaluator(
                              rows, device=device, batch_size=batch_size, cache_dir=cache_dir)}
                verify_semantic(report, rows, sources[arm]["sha256"],
                                device=device, batch_size=batch_size)
                if any(sha256_file(Path(s["path"])) != s["sha256"] for s in sources.values()):
                    raise ValueError("Validation inputs changed during semantic scoring")
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(report, handle, ensure_ascii=False, indent=2)
            verify_semantic(report, rows, sources[arm]["sha256"],
                            device=device, batch_size=batch_size)
            reports[arm] = report
        for arm in ARMS:
            compare_bertscore_reports(
                reports["m0"], reports[arm],
                left_predictions_sha256=sources["m0"]["sha256"],
                right_predictions_sha256=sources[arm]["sha256"], expected_split="validation",
                expected_question_images={r["question_id"]: r["image_id"]
                                          for r in predictions[arm]})
        if any(sha256_file(Path(s["path"])) != s["sha256"] for s in sources.values()):
            raise ValueError("Validation inputs changed during semantic scoring")
        result = {"status": "completed", "questions_per_arm": 1407,
                  "scores": {a: r["explanation_bertscore"]["mean_f1"]
                             for a, r in reports.items()},
                  "files": {f"{a}_metrics.json": sha256_file(output / f"{a}_metrics.json")
                            for a in ARMS}}
        receipt = output / "completed.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text("utf-8"))
            if any(previous.get(key) != value for key, value in result.items()):
                raise ValueError("Completed semantic receipt changed")
        else:
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump({**result, "completed_at_utc": utc_now()}, handle, indent=2)
        return result
