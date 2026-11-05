"""Verify eight final-test exports and summarize the fixed comparisons."""

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, stdev

from vqa_explanation.data.io import write_json
from vqa_explanation.evaluation.answer_metrics import exact_match, token_f1
from vqa_explanation.evaluation.comparison import cluster_bootstrap, compare_predictions
from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1
from vqa_explanation.evaluation.rubric_calibration import check_new_output
from vqa_explanation.evaluation.semantic_metrics import compare_bertscore_reports
from vqa_explanation.inference.export import read_jsonl, select_records, validate_saved_predictions
from vqa_explanation.inference.study_test import (
    IDENTITIES,
    checked_protocol,
    read_test_plan,
    verify_test_completion,
)
from vqa_explanation.utils.run_provenance import sha256_file


def inspect_test_outputs(plan_path, plan_sha256, annotations_path, runs):
    plan = read_test_plan(plan_path, plan_sha256)
    if set(runs) != set(IDENTITIES):
        raise ValueError("Provide exactly the eight planned test identities")
    annotations_path = Path(annotations_path)
    if sha256_file(annotations_path) != plan["test_annotations_sha256"]:
        raise ValueError("Frozen test annotations changed")
    annotations = read_jsonl(annotations_path)
    if (len(annotations) != 1968 or len({r["question_id"] for r in annotations}) != 1968
            or any(r["split"] != "test" for r in annotations)
            or len({r["image_id"] for r in annotations}) != plan["test_images"]):
        raise ValueError("Require all frozen test questions and images")
    selected = select_records(annotations, "test", 0, 42)
    selected_hash = hashlib.sha256(json.dumps(
        selected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    sources = {"plan": {"path": str(Path(plan_path).resolve()), "sha256": plan_sha256},
               "annotations": {"path": str(annotations_path.resolve()),
                               "sha256": sha256_file(annotations_path)}}
    predictions, progress, shared_runtime, rubric_hash = {}, {}, None, None
    for identity in IDENTITIES:
        directory = Path(runs[identity]).resolve()
        output = directory / f"{identity}.jsonl"
        manifest_path = output.with_suffix(".jsonl.manifest.json")
        receipt = directory / "test_run.json"
        completion = verify_test_completion(directory, identity)
        if not receipt.exists():
            if output.exists() or manifest_path.exists() or completion is not None:
                raise ValueError(f"{identity}: export has no test run record")
            progress[identity] = {"status": "missing", "saved": 0, "expected": 1968}
            continue
        record = json.loads(receipt.read_text("utf-8"))
        expected = plan["runs"][identity]
        if (record.get("schema") != "final_test_run_v1"
                or record.get("test_identity") != identity
                or record.get("training_seed") != expected["training_seed"]
                or record.get("plan_sha256") != plan_sha256
                or record.get("adapter_files") != expected["adapter_files"]
                or not re.fullmatch(r"[0-9a-f]{64}", record.get("rubric_lock_sha256", ""))):
            raise ValueError(f"{identity}: test run record differs from the frozen plan")
        if rubric_hash is None:
            rubric_hash = record["rubric_lock_sha256"]
        if record["rubric_lock_sha256"] != rubric_hash:
            raise ValueError("Test identities use different frozen rubrics")
        for name, path in (("run", receipt), ("manifest", manifest_path), ("predictions", output),
                           ("completion", directory / "completion.json")):
            if path.is_file():
                sources[f"{identity}_{name}"] = {"path": str(path), "sha256": sha256_file(path)}
        if not manifest_path.exists():
            if output.exists():
                raise ValueError(f"{identity}: predictions have no export manifest")
            progress[identity] = {"status": "prepared", "saved": 0, "expected": 1968}
            continue
        manifest = json.loads(manifest_path.read_text("utf-8"))
        inputs = manifest["inputs"]
        context = {k: record[k] for k in (
            "test_identity", "training_seed", "plan_sha256", "rubric_lock_sha256")}
        if (manifest.get("schema_version") != 1 or inputs.get("context") != context
                or inputs.get("adapter") != record.get("adapter")
                or inputs.get("adapter_files") != expected["adapter_files"]
                or inputs.get("source_files") != plan["inference_source"]
                or inputs.get("experiment_id") != expected["experiment_id"]
                or inputs.get("annotations_sha256") != plan["test_annotations_sha256"]
                or inputs.get("selected_sha256") != selected_hash
                or inputs.get("split") != "test" or inputs.get("limit") != 0
                or any(inputs.get(k) != plan["inference_protocol"][k] for k in (
                    "model_id", "seed", "min_pixels", "max_pixels", "max_new_tokens",
                    "load_in_4bit"))
                or inputs.get("model_revision") != plan["inference_protocol"]["model_revision"]):
            raise ValueError(f"{identity}: export manifest differs from the frozen test identity")
        checked_protocol(manifest["protocol"], plan["inference_protocol"])
        runtime = {key: inputs.get(key) for key in ("python", "versions")}
        if not runtime["python"] or not isinstance(runtime["versions"], dict):
            raise ValueError(f"{identity}: missing inference runtime metadata")
        if shared_runtime is None:
            shared_runtime = runtime
        if shared_runtime != runtime:
            raise ValueError("Test inference runtime differs between identities")
        rows = read_jsonl(output) if output.exists() else []
        validate_saved_predictions(rows, selected, manifest)
        if output.exists() and output.stat().st_size and not output.read_bytes().endswith(b"\n"):
            raise ValueError(f"{identity}: incomplete final line; inspect its recovery snapshots")
        complete = completion is not None and len(rows) == 1968
        progress[identity] = {"status": "completed" if complete else "incomplete",
                              "saved": len(rows), "expected": 1968}
        if complete:
            predictions[identity] = rows
    return plan, annotations, predictions, progress, sources


def analyze_test_outputs(plan, annotations, predictions, sources, semantic=None):
    if set(predictions) != set(IDENTITIES):
        raise ValueError("No scoring until all eight test exports are complete")
    canonical = {r["question_id"]: r for r in annotations}
    if (len(annotations) != 1968 or len(canonical) != 1968
            or any(r["split"] != "test" for r in annotations)):
        raise ValueError("Scoring requires the complete canonical test split")
    ids = sorted(canonical)
    images = [canonical[q]["image_id"] for q in ids]
    pairs = [("m0", "m1")] + [(left, right) for seed in (42, 43, 44) for left, right in (
        ("m1", f"m2_seed{seed}"), ("m1", f"m3_seed{seed}"),
        (f"m2_seed{seed}", f"m3_seed{seed}"))]
    for left, right in pairs:
        checked = compare_predictions(
            predictions[left], predictions[right], expected_annotations=canonical,
            expected_split="test", expected_experiments=(plan["runs"][left]["experiment_id"],
                                                         plan["runs"][right]["experiment_id"]))
        if not checked["comparable"] or not checked["explanation_rouge_l_f1"]["comparable"]:
            raise ValueError(f"{left}/{right}: incompatible test exports")
    values, diagnostics = {}, {}
    for identity, rows in predictions.items():
        indexed = {r["question_id"]: r for r in rows}
        ordered = [indexed[q] for q in ids]
        values[identity] = {
            "exact_match": [exact_match(r["prediction_answer"], r["reference_answer"])
                            for r in ordered],
            "token_f1": [token_f1(r["prediction_answer"], r["reference_answer"]) for r in ordered],
            "rouge_l": [rouge_l_f1(r["prediction_explanation"], r["reference_explanations"])
                        for r in ordered]}
        diagnostics[identity] = {"format_valid_n": sum(r["format_valid"] for r in ordered),
                                 "empty_answer_n": sum(not r["prediction_answer"].strip()
                                                       for r in ordered),
                                 "empty_explanation_n": sum(not r["prediction_explanation"].strip()
                                                            for r in ordered)}
    semantic_protocol = None
    if semantic is not None:
        if set(semantic) != set(IDENTITIES):
            raise ValueError("Provide BERTScore for all eight identities or none")
        for identity, report in semantic.items():
            checked, _, _ = compare_bertscore_reports(
                semantic["m0"], report, expected_split="test",
                left_predictions_sha256=sources["m0_predictions"]["sha256"],
                right_predictions_sha256=sources[f"{identity}_predictions"]["sha256"],
                expected_question_images=dict(zip(ids, images, strict=True)))
            semantic_protocol = checked["protocol"]
            if any(semantic_protocol.get(key) != expected for key, expected in (
                ("model_id", plan["metrics"]["bertscore_model"]),
                ("model_revision", plan["metrics"]["bertscore_revision"]),
                ("num_layers", plan["metrics"]["bertscore_layers"]), ("idf", False),
                ("rescale_with_baseline", False), ("use_fast_tokenizer", False),
                ("reference_aggregation", "best_match_across_all_references")
            )):
                raise ValueError("BERTScore protocol differs from the frozen test plan")
            indexed = {r["question_id"]: r["f1"] for r in report["explanation_bertscore"]["items"]}
            values[identity]["bertscore_f1"] = [indexed[q] for q in ids]
    metrics = list(values["m0"])
    scores = {identity: {name: mean(items) for name, items in row.items()}
              for identity, row in values.items()}
    bootstrap = {}
    for left, right in pairs:
        bootstrap[f"{right}_minus_{left}"] = cluster_bootstrap(
            {name: [b - a for a, b in zip(values[left][name], values[right][name], strict=True)]
             for name in metrics}, images, seed=plan["metrics"]["bootstrap_seed"],
            resamples=plan["metrics"]["bootstrap_resamples"])
    across_seeds = {}
    for name in ("m2", "m3", "m3_minus_m2"):
        across_seeds[name] = {}
        for metric in metrics:
            items = [scores[f"{name}_seed{seed}"][metric] if name != "m3_minus_m2" else
                     scores[f"m3_seed{seed}"][metric] - scores[f"m2_seed{seed}"][metric]
                     for seed in (42, 43, 44)]
            across_seeds[name][metric] = {"mean": mean(items), "sample_sd": stdev(items),
                                         "values_by_seed": dict(zip((42, 43, 44), items,
                                                                    strict=True))}
    return {"schema": "final_test_report_v1", "split": "test", "n": len(ids),
            "n_images": len(set(images)), "status": "automatic_metrics_complete" if semantic
            else "lexical_complete_semantic_pending", "human_evaluation_complete": False,
            "human_calibration_complete_before_inference": (
                False if "sequence_amendment" in plan else None),
            "evaluation_sequence_amendment": plan.get("sequence_amendment"),
            "scores": scores, "diagnostics": diagnostics, "bootstrap": bootstrap,
            "training_seeds": [42, 43, 44], "inference_order_seed": 42,
            "fixed_baselines": {key: scores[key] for key in ("m0", "m1")},
            "seed_summary": across_seeds, "semantic_protocol": semantic_protocol,
            "inference_protocol": plan["inference_protocol"], "sources": sources,
            "limitations": ["Fixed M0/M1 are each one export, not three independent runs.",
                            "Text metrics do not measure visual support or replace human ratings.",
                            "This report checks recorded image hashes, not model loading on GPU.",
                            "Do not use test results to select a seed, checkpoint, prompt or LR."]}


def build_test_report(plan_path, plan_sha256, annotations_path, runs, *, semantic_paths=None,
                      output_dir=None, inspect_only=False):
    plan, annotations, predictions, progress, sources = inspect_test_outputs(
        plan_path, plan_sha256, annotations_path, runs)
    if inspect_only:
        return {"status": "ready_to_score" if len(predictions) == 8 else "awaiting_exports",
                "progress": progress, "metrics_computed": False,
                "human_calibration_complete_before_inference": (
                    False if "sequence_amendment" in plan else None),
                "human_evaluation_complete": False}
    if len(predictions) != 8:
        raise ValueError("No scoring until all eight test exports are complete")
    if output_dir is None:
        raise ValueError("Scoring requires a new output directory")
    semantic = None
    if semantic_paths is not None:
        if set(semantic_paths) != set(IDENTITIES):
            raise ValueError("Provide exactly eight semantic reports")
        semantic = {name: json.loads(Path(path).read_text("utf-8"))
                    for name, path in semantic_paths.items()}
        sources.update({f"{name}_semantic": {"path": str(Path(path).resolve()),
                                            "sha256": sha256_file(Path(path))}
                        for name, path in semantic_paths.items()})
    output = check_new_output(output_dir, [*runs.values(),
                                          *(s["path"] for s in sources.values())])
    report = analyze_test_outputs(plan, annotations, predictions, sources, semantic)
    if any(sha256_file(Path(s["path"])) != s["sha256"] for s in sources.values()):
        raise ValueError("Test inputs changed during scoring")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    metrics = list(report["scores"]["m0"])
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["identity", *metrics])
        writer.writerows([name, *(row[key] for key in metrics)]
                         for name, row in report["scores"].items())
    (output / "README.md").write_text(
        "# Kết quả test\n\n1.968 câu cho mỗi identity; M0/M1 giữ một lần xuất. "
        "CSV dùng giá trị gốc, nhân 100 khi trình bày. `report.json` chứa CI theo ảnh "
        "và mean/SD ba seed, không lấy mẫu số bằng số câu nhân ba.\n\n"
        + ("BERTScore đã đối chiếu đủ tám identity. " if semantic else "BERTScore còn thiếu. ")
        + "Chấm theo ảnh vẫn là phần riêng; bảng này không kết luận về độ bám ảnh.\n", "utf-8")
    write_json(output / "manifest.json", {"files": {
        p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    return report
