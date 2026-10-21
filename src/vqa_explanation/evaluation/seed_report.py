"""Summarize the three declared training seeds without selecting a winner."""

import json
import math
from pathlib import Path
from statistics import mean, stdev

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.evaluation.study_report import METRICS, analyze_study, verify_evidence
from vqa_explanation.utils.run_provenance import sha256_file

PLANNED_SEEDS = (42, 43, 44)
ARMS = ("m0", "m1", "m2", "m3")
BASELINE_BERTSCORE_ABS_TOLERANCE = 1e-7


def summarize_seeds(reports):
    if set(reports) != set(PLANNED_SEEDS):
        raise ValueError("Require exactly the declared training seeds 42, 43 and 44")
    baseline = reports[42]
    for seed, report in reports.items():
        if (report.get("schema") != "four_arm_validation_report_v1"
                or report.get("training_seed") != seed or report.get("split") != "validation"
                or report.get("n") != 1407 or report.get("n_images") != 1383
                or set(report.get("scores", {})) != set(ARMS)
                or not report.get("semantic_protocol")):
            raise ValueError(f"Seed {seed}: incomplete four-arm validation report")
        for key in ("inference_protocol", "semantic_protocol"):
            if report[key] != baseline[key]:
                raise ValueError(f"Seed {seed}: different {key}")
        for key in ("annotations", "canonical_all", "m1_selection", "m0", "m1"):
            if report["sources"][key]["sha256"] != baseline["sources"][key]["sha256"]:
                raise ValueError(f"Seed {seed}: different {key}")
        for arm in ARMS:
            scores = report["scores"][arm]
            if (set(scores) != set(METRICS)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in scores.values())):
                raise ValueError(f"Seed {seed}/{arm}: missing or non-finite metric")
            if arm in ("m0", "m1"):
                for metric, value in scores.items():
                    tolerance = (BASELINE_BERTSCORE_ABS_TOLERANCE
                                 if metric == "bertscore_f1" else 0)
                    if not math.isclose(value, baseline["scores"][arm][metric],
                                        rel_tol=0, abs_tol=tolerance):
                        raise ValueError("Fixed baselines disagree; do not average "
                                         "repeated baselines")
        for arm in ("m2", "m3"):
            training = report["evidence"]["training"][arm]
            if (training.get("status") != "passed" or training.get("global_step") != 375
                    or training.get("epoch") != 3):
                raise ValueError(f"Seed {seed}/{arm}: incomplete training evidence")
    summaries = {}
    for arm in ("m2", "m3", "m3_minus_m2"):
        summaries[arm] = {}
        for metric in METRICS:
            values = [reports[s]["scores"][arm][metric] if arm != "m3_minus_m2" else
                      reports[s]["scores"]["m3"][metric] - reports[s]["scores"]["m2"][metric]
                      for s in PLANNED_SEEDS]
            summaries[arm][metric] = {"mean": mean(values), "sample_sd": stdev(values),
                                      "values_by_seed": dict(zip(PLANNED_SEEDS, values,
                                                                 strict=True))}
    return {
        "schema": "validation_seed_summary_v1", "split": "validation",
        "training_seeds": list(PLANNED_SEEDS), "n": 1407, "n_images": 1383,
        "fixed_baselines": {a: baseline["scores"][a] for a in ("m0", "m1")},
        "baseline_bertscore_abs_tolerance": BASELINE_BERTSCORE_ABS_TOLERANCE,
        "summary": summaries,
        "per_seed_m3_minus_m2_ci95": {
            s: reports[s]["bootstrap"]["m3_minus_m2"] for s in PLANNED_SEEDS},
        "inference_protocol": baseline["inference_protocol"],
        "semantic_protocol": baseline["semantic_protocol"],
        "limitations": "Sample SD describes three preference-training seeds from one fixed "
                        "M1. Image-bootstrap CIs remain separate per seed; they are not "
                        "averaged or treated as seed-level confidence intervals. Text metrics "
                        "do not establish visual grounding. No test results are used.",
    }


def render_seed_markdown(report):
    lines = ["# Validation qua ba seed", "",
             "M2/M3 lặp với seed 42, 43, 44 từ cùng M1. M0/M1 là baseline cố định, "
             "không phải ba lần train độc lập. Các điểm nhân 100.", "",
             "| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore F1 |",
             "|---|---:|---:|---:|---:|"]
    for arm, scores in report["fixed_baselines"].items():
        lines.append(f"| {arm.upper()} | " + " | ".join(
            f"{scores[m] * 100:.2f}" for m in METRICS) + " |")
    for arm in ("m2", "m3", "m3_minus_m2"):
        scores = report["summary"][arm]
        label = "M3 − M2" if arm == "m3_minus_m2" else arm.upper()
        lines.append(f"| {label} | " + " | ".join(
            f"{scores[m]['mean'] * 100:.2f} ± {scores[m]['sample_sd'] * 100:.2f}"
            for m in METRICS) + " |")
    lines += ["", "± là độ lệch chuẩn mẫu giữa ba seed, không phải CI 95%.", "",
              "## Chênh lệch M3 − M2 từng seed", "",
              "| Seed | EM | Token-F1 | ROUGE-L | BERTScore F1 |",
              "|---|---:|---:|---:|---:|"]
    for seed in report["training_seeds"]:
        lines.append(f"| {seed} | " + " | ".join(
            f"{report['summary']['m3_minus_m2'][m]['values_by_seed'][seed] * 100:+.2f}"
            for m in METRICS) + " |")
    lines += ["", "CI bootstrap theo ảnh của từng seed được giữ riêng trong `report.json`. "
              "Không chọn seed có điểm cao nhất. Mức giống văn bản tham chiếu chưa chứng minh "
              "giải thích bám ảnh; test và chấm mù là các bước khác.", ""]
    return "\n".join(lines)


def build_seed_report(paths, output_dir):
    """Require intact per-seed reports, their manifests and original input files."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    reports, sources = {}, {}
    for seed, filename in sorted(paths.items()):
        path = Path(filename)
        manifest = json.loads((path.parent / "manifest.json").read_text("utf-8"))
        if manifest["files"].get(path.name) != sha256_file(path):
            raise ValueError(f"Seed {seed}: report differs from its saved manifest")
        report = json.loads(path.read_text("utf-8"))
        for source in report["sources"].values():
            if sha256_file(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"Seed {seed}: original report input changed")
        inputs = report["sources"]
        predictions = {a: list(read_jsonl(inputs[a]["path"])) for a in ARMS}
        annotations = list(read_jsonl(inputs["annotations"]["path"]))
        evidence = verify_evidence(
            Path(inputs["m0"]["path"]).parent.parent, annotations, predictions,
            Path(inputs["study_config"]["path"]), Path(inputs["canonical_all"]["path"]))
        semantic = {a: json.loads(Path(inputs[f"{a}_semantic"]["path"]).read_text("utf-8"))
                    for a in ARMS}
        recomputed = analyze_study(
            predictions, annotations, semantic=semantic,
            prediction_hashes={a: inputs[a]["sha256"] for a in ARMS},
            training_seed=evidence["training_seed"], resamples=2)
        if any(recomputed[key] != report[key] for key in (
            "scores", "training_seed", "inference_protocol", "semantic_protocol", "n", "n_images"
        )):
            raise ValueError(f"Seed {seed}: report does not match its source data")
        for arm in ("m2", "m3"):
            plan = json.loads(Path(inputs[f"training/{arm}/data_plan.json"]["path"]).read_text(
                "utf-8"))
            identity = {k: plan[k] for k in (
                "preferences_manifest_sha256", "preferences_files", "m1_adapter",
                "m1_selection_sha256", "source_files", "visual_quality", "model")}
            identity["config"] = {k: v for k, v in plan["config"].items() if k != "seed"}
            if seed == 42:
                sources[f"{arm}_training_identity"] = identity
            elif identity != sources[f"{arm}_training_identity"]:
                raise ValueError(f"Seed {seed}/{arm}: training design changed beyond seed")
        reports[seed] = report
        sources[seed] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    result = summarize_seeds(reports)
    result["sources"] = {s: sources[s] for s in PLANNED_SEEDS}
    output.mkdir(parents=True)
    (output / "README.md").write_text(render_seed_markdown(result), "utf-8")
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    (output / "manifest.json").write_text(json.dumps({"files": {
        p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}, indent=2), "utf-8")
    return result
