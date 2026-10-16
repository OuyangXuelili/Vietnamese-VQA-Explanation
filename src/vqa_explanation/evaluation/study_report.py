"""Compare four saved study arms without generating or selecting new predictions."""

import hashlib
import json
from collections import Counter
from html import escape
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageOps

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.answer_metrics import exact_match, token_f1
from vqa_explanation.evaluation.comparison import (
    PROTOCOL_FIELDS,
    cluster_bootstrap,
    compare_predictions,
)
from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
    compare_bertscore_reports,
)
from vqa_explanation.inference.export import select_records, validate_saved_predictions
from vqa_explanation.inference.study_validation import (
    ARMS,
    MODEL_ID,
    MODEL_REVISION,
    VALIDATION_SHA256,
    main_study_settings,
    verify_main_stage,
)
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file

METRICS = {"exact_match": "EM", "token_f1": "Token-F1", "rouge_l": "ROUGE-L",
           "bertscore_f1": "BERTScore F1"}
GROUPS = {"both_match": "M2 và M3 cùng khớp", "gain": "M2 không khớp → M3 khớp",
          "loss": "M2 khớp → M3 không khớp", "neither_match": "Cả hai không khớp"}


def analyze_study(predictions, annotations, *, semantic=None, prediction_hashes=None,
                  seed=42, resamples=2000, per_group=8):
    if set(predictions) != set(ARMS):
        raise ValueError("Require exactly M0, M1, M2 and M3 predictions")
    if type(per_group) is not int or per_group < 1:
        raise ValueError("Cases per group must be a positive integer")
    canonical = {r["question_id"]: r for r in annotations}
    if (not canonical or len(canonical) != len(annotations)
            or any(r["split"] != "validation" for r in annotations)):
        raise ValueError("Require unique validation annotations")
    ids = sorted(canonical)
    images = [canonical[q]["image_id"] for q in ids]
    indexed = {arm: {r["question_id"]: r for r in rows} for arm, rows in predictions.items()}
    comparisons = {}
    for left, right in combinations(ARMS, 2):
        comparison = compare_predictions(
            predictions[left], predictions[right], expected_annotations=canonical,
            expected_experiments=(ARMS[left], ARMS[right]))
        if (not comparison["comparable"]
                or not comparison["explanation_rouge_l_f1"]["comparable"]):
            raise ValueError(f"{left}/{right}: incompatible predictions: {comparison['issues']}")
        comparisons[f"{right}_minus_{left}"] = comparison
    values, diagnostics = {}, {}
    for arm in ARMS:
        rows = [indexed[arm][q] for q in ids]
        if any(type(r.get("format_valid")) is not bool for r in rows):
            raise ValueError("Exporter format flags are required on every row")
        values[arm] = {
            "exact_match": [exact_match(r["prediction_answer"], r["reference_answer"])
                            for r in rows],
            "token_f1": [token_f1(r["prediction_answer"], r["reference_answer"]) for r in rows],
            "rouge_l": [rouge_l_f1(r["prediction_explanation"], r["reference_explanations"])
                        for r in rows],
        }
        diagnostics[arm] = {
            "format_valid_n": sum(r["format_valid"] for r in rows),
            "empty_answer_n": sum(not r["prediction_answer"].strip() for r in rows),
            "empty_explanation_n": sum(not r["prediction_explanation"].strip() for r in rows),
            "mean_explanation_whitespace_tokens": mean(
                len(r["prediction_explanation"].split()) for r in rows),
            "distinct_explanations": len({r["prediction_explanation"].strip() for r in rows}),
            "median_latency_seconds": median(r["latency_seconds"] for r in rows),
        }
    semantic_protocol = None
    if semantic is not None:
        if set(semantic) != set(ARMS) or set(prediction_hashes or {}) != set(ARMS):
            raise ValueError("Require four semantic reports and their prediction hashes")
        for arm in ARMS:
            comparison, _, _ = compare_bertscore_reports(
                semantic["m0"], semantic[arm],
                left_predictions_sha256=prediction_hashes["m0"],
                right_predictions_sha256=prediction_hashes[arm], expected_split="validation",
                expected_question_images=dict(zip(ids, images, strict=True)))
            semantic_protocol = comparison["protocol"]
            items = {r["question_id"]: r for r in semantic[arm]["explanation_bertscore"]["items"]}
            values[arm]["bertscore_f1"] = [items[q]["f1"] for q in ids]
    bootstrap = {}
    for left, right in combinations(ARMS, 2):
        differences = {name: [b - a for a, b in zip(values[left][name], values[right][name],
                                                   strict=True)] for name in values[left]}
        bootstrap[f"{right}_minus_{left}"] = cluster_bootstrap(
            differences, images, seed=seed, resamples=resamples)
    groups = {name: [] for name in GROUPS}
    for i, qid in enumerate(ids):
        a, b = values["m2"]["exact_match"][i], values["m3"]["exact_match"][i]
        group = "both_match" if a and b else "loss" if a else "gain" if b else "neither_match"
        groups[group].append(qid)
    selected = {}
    for group, qids in groups.items():
        ordered = sorted(qids, key=lambda q: (
            hashlib.sha256(f"study_validation_case_v1:{seed}:{group}:{q}".encode()).hexdigest(),
            q))
        seen, selected[group] = set(), []
        for qid in ordered:
            if canonical[qid]["image_id"] not in seen:
                selected[group].append(qid)
                seen.add(canonical[qid]["image_id"])
            if len(selected[group]) == per_group:
                break
    return {
        "schema": "four_arm_validation_report_v1", "split": "validation", "n": len(ids),
        "n_images": len(set(images)), "training_seed": seed,
        "scores": {arm: {k: mean(v) for k, v in scores.items()}
                   for arm, scores in values.items()},
        "comparisons": comparisons, "bootstrap": bootstrap, "diagnostics": diagnostics,
        "semantic_protocol": semantic_protocol,
        "em_transitions_m2_m3": {g: len(q) for g, q in groups.items()},
        "case_selection": {"seed": seed, "per_group": per_group, "ids": selected,
                           "method": "SHA-256 rank within M2/M3 EM transition; one question "
                                     "per image within each group; no visual judgement"},
        "inference_protocol": {k: predictions["m0"][0][k] for k in PROTOCOL_FIELDS},
        "adapters": {arm: predictions[arm][0]["adapter"] for arm in ARMS},
    }


def verify_evidence(folder, annotations, predictions, study_config, canonical_all_path):
    """Check frozen training records and export manifests, not just notebook text."""
    folder = Path(folder)
    settings = main_study_settings(study_config)
    plans, reports, manifests = {}, {}, {}
    canonical_hash = sha256_file(canonical_all_path)
    for arm in ("m2", "m3"):
        stage = folder / "training" / arm
        report = json.loads((stage / "training_result.json").read_text("utf-8"))
        plan = json.loads((stage / "data_plan.json").read_text("utf-8"))
        digest = hashlib.sha256(json.dumps(
            {k: v for k, v in plan.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()
        if (digest != plan["sha256"] or report.get("plan_sha256") != digest
                or plan.get("pairs") != settings[arm]["pairs"]
                or report.get("status") != "passed" or report.get("global_step") != 375
                or report.get("epoch") != 3 or report.get("policy_updated") is not True
                or report.get("saved_policy_verified") is not True
                or (arm == "m3" and report.get("reference_unchanged") is not True)
                or plan.get("model", {}).get("id") != MODEL_ID
                or plan.get("model", {}).get("revision") != MODEL_REVISION):
            raise ValueError(f"{arm}: incomplete or changed training evidence")
        verify_main_stage(stage, arm, report, plan, settings[arm], canonical_hash)
        plans[arm], reports[arm] = plan, report
    for key in ("annotations_sha256", "preferences_manifest_sha256", "preferences_files",
                "m1_adapter", "m1_selection_sha256", "source_files", "visual_quality"):
        if plans["m2"][key] != plans["m3"][key]:
            raise ValueError(f"Training arms disagree on {key}")
    if sha256_file(folder / "m1_selection.json") != plans["m2"]["m1_selection_sha256"]:
        raise ValueError("M1 selection changed")
    selected = select_records(annotations, "validation", 0, 42)
    selected_hash = hashlib.sha256(json.dumps(
        selected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    for arm in ARMS:
        manifest = json.loads((folder / "validation" / f"{arm}.jsonl.manifest.json").read_text(
            "utf-8"))
        inputs = manifest["inputs"]
        expected_files = (None if arm == "m0" else plans["m2"]["m1_adapter"]["files"]
                          if arm == "m1" else reports[arm]["output_adapter"]["files"])
        if (manifest.get("schema_version") != 1 or inputs.get("model_id") != MODEL_ID
                or inputs.get("model_revision") != MODEL_REVISION
                or inputs.get("experiment_id") != ARMS[arm]
                or inputs.get("adapter_files") != expected_files
                or inputs.get("annotations_sha256") != VALIDATION_SHA256
                or inputs.get("selected_sha256") != selected_hash
                or inputs.get("split") != "validation" or inputs.get("limit") != 0
                or inputs.get("seed") != 42
                or inputs.get("min_pixels") != 200704 or inputs.get("max_pixels") != 200704
                or inputs.get("max_new_tokens") != 128 or inputs.get("load_in_4bit") is not True
                or manifest["protocol"].get("generation_config", {}).get("do_sample") is not False
                or manifest["protocol"].get("generation_config", {}).get("num_beams") != 1):
            raise ValueError(f"{arm}: export identity does not match the locked study")
        completed = validate_saved_predictions(predictions[arm], selected, manifest)
        if len(completed) != len(selected):
            raise ValueError(f"{arm}: incomplete export")
        manifests[arm] = manifest
    for arm in ARMS:
        for field in ("source_files", "python", "versions"):
            if manifests[arm]["inputs"][field] != manifests["m0"]["inputs"][field]:
                raise ValueError(f"{arm}: inference {field} differs")
    for name, digest in manifests["m0"]["inputs"]["source_files"].items():
        path = (project_root() / name).resolve()
        if not path.is_relative_to(project_root()) or sha256_file(path) != digest:
            raise ValueError("Inference source differs from the local implementation")
    return {"manifests": manifests, "training": reports,
            "checkpoint_files_verified_by": "completed export manifests and saved train reports; "
                                            "weights were checked before inference"}


def render_study_markdown(report):
    lines = ["# Validation M0–M3 — seed 42", "",
             f"{report['n']:,} câu / {report['n_images']:,} ảnh. "
             "M2 và M3 cùng khởi đầu từ M1-3683, học 1.000 cặp trong ba epoch (375 update). "
             "Đây là kết quả validation của một seed, chưa phải kết quả test.", "",
             "| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore F1 |", "|---|---:|---:|---:|---:|"]
    for arm, scores in report["scores"].items():
        values = [f"{scores[m] * 100:.2f}" if m in scores else "Chưa đo" for m in METRICS]
        lines.append(f"| {arm.upper()} | " + " | ".join(values) + " |")
    lines += ["", "Các chỉ số trong bảng nhân 100. Giữ mọi output trong mẫu số, "
              "kể cả sai định dạng. M0/M1 là baseline; đối chứng chính là M3 − M2.", "",
              "![Điểm validation](metrics.png)", "", "## Chênh lệch M3 − M2", "",
              "| Chỉ số | Chênh lệch | CI 95% |", "|---|---:|---:|"]
    for metric, stat in report["bootstrap"]["m3_minus_m2"]["metrics"].items():
        lo, hi = (v * 100 for v in stat["ci95"])
        lines.append(f"| {METRICS[metric]} | {stat['delta_right_minus_left'] * 100:+.2f} "
                     f"| [{lo:+.2f}; {hi:+.2f}] |")
    lines += ["", "![Chênh lệch và CI](deltas.png)", "",
              "CI dùng paired image-bootstrap 2.000 lần, seed 42; đo bất định tập ảnh, "
              "không thay độ lệch chuẩn qua các seed train. Các CI là khoảng riêng từng "
              "chỉ số, không phải kiểm định đồng thời đã hiệu chỉnh nhiều phép so sánh.", "",
              "## Định dạng và độ dài", "",
              "| Nhánh | Đúng định dạng | Đáp án rỗng | Giải thích rỗng | "
              "Độ dài giải thích TB |", "|---|---:|---:|---:|---:|"]
    for arm, d in report["diagnostics"].items():
        lines.append(f"| {arm.upper()} | {d['format_valid_n']}/{report['n']} | "
                     f"{d['empty_answer_n']} | {d['empty_explanation_n']} | "
                     f"{d['mean_explanation_whitespace_tokens']:.2f} |")
    lines += ["", "Độ dài tính bằng tách khoảng trắng, không phải token Qwen. "
              "Không dùng latency từ những phiên bị nối lại để so tốc độ mô hình.", "",
              "## Ví dụ cố định", "",
              "| Nhóm M2/M3 theo EM | Số câu toàn tập |", "|---|---:|"]
    lines += [f"| {GROUPS[g]} | {n} |" for g, n in report["em_transitions_m2_m3"].items()]
    lines += ["", "Mở [cases.html](cases.html) để xem ảnh và cả bốn đầu ra. "
              "Chọn tối đa tám ảnh mỗi nhóm bằng hash, giữ cả trường hợp giảm điểm. "
              "Không khớp tham chiếu chưa chắc sai theo ảnh; bảng nhóm không phải nhãn chấm mù.",
              "", "## Giới hạn và bước tiếp", "",
              "ROUGE-L và BERTScore đo tương đồng văn bản, chưa chứng minh giải thích "
              "được ảnh hỗ trợ. Cần đối chứng đổi ảnh, seed 43/44, test sau khóa thiết kế "
              "và chấm mù theo rubric. Không chọn checkpoint hoặc seed đẹp nhất từ bảng này.",
              "", "Các input, hash, giao thức, hồ sơ train và sáu phép so sánh đầy đủ "
              "nằm trong `report.json`. Không sửa prediction hoặc output train khi lập báo cáo."]
    return "\n".join(lines) + "\n"


def render_study_cases(report, indexed, canonical):
    parts = ["<!doctype html><html lang='vi'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Validation M0–M3</title><style>body{font:16px/1.55 Arial;max-width:1200px;"
             "margin:auto;padding:24px;color:#243442}article{border-top:1px solid #ccc;"
             "padding:24px 0}img{max-width:100%;height:300px;object-fit:contain}.outputs{"
             "display:grid;grid-template-columns:1fr 1fr;gap:20px}p{white-space:pre-wrap}"
             "section{background:#f4f6f8;padding:12px}@media(max-width:700px){.outputs{"
             "grid-template-columns:1fr}}</style><h1>Validation M0–M3</h1><p>"
             "Ví dụ chọn theo quy tắc cố định, không phải chấm mù hoặc tỷ lệ lỗi toàn tập. "
             "EM chỉ đo khớp tham chiếu; ảnh và tham chiếu cần được đọc cùng nhau.</p>"]
    for group, ids in report["case_selection"]["ids"].items():
        parts.append(f"<h2>{GROUPS[group]}</h2>")
        for qid in ids:
            ref = canonical[qid]
            parts.append(f"<article id='q{escape(qid, quote=True)}'><h3>"
                         f"{escape(ref['question'])}</h3><p>Câu {escape(qid)} · "
                         f"ảnh {escape(ref['image_id'])}</p><img src='images/"
                         f"{escape(qid, quote=True)}.jpg' alt='Ảnh câu hỏi'><div class='outputs'>")
            for arm in ARMS:
                row = indexed[arm][qid]
                em = exact_match(row["prediction_answer"], ref["answer"])
                parts.append(f"<section><h4>{arm.upper()} · EM {int(em)}</h4><p>"
                             f"Đáp án: {escape(row['prediction_answer'])}</p><p>"
                             f"Giải thích: {escape(row['prediction_explanation'])}</p></section>")
            refs = "".join(f"<li>{escape(e)}</li>" for e in ref["explanations"])
            parts.append(f"</div><details><summary>Tham chiếu: {escape(ref['answer'])}"
                         f"</summary><ul>{refs}</ul></details></article>")
    return "\n".join(parts) + "</html>"


def save_study_figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(report["scores"]["m0"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for i, arm in enumerate(ARMS):
        bars = ax.bar(x + (i - 1.5) * .2,
                      [report["scores"][arm][m] * 100 for m in names], .19, label=arm.upper())
        ax.bar_label(bars, fmt="%.2f", fontsize=8, rotation=90, padding=3)
    ax.set(xticks=x, xticklabels=[METRICS[n] for n in names], ylim=(0, 105),
           ylabel="Score × 100", title=f"Validation · {report['n']} questions · training seed 42")
    ax.legend(frameon=False, ncols=4)
    fig.savefig(output / "metrics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharey=True)
    for ax, pair in zip(axes, ("m2_minus_m1", "m3_minus_m1", "m3_minus_m2"), strict=True):
        for i, metric in enumerate(names):
            stat = report["bootstrap"][pair]["metrics"][metric]
            ax.plot([v * 100 for v in stat["ci95"]], [i, i], linewidth=3, color="#087f8c")
            ax.scatter(stat["delta_right_minus_left"] * 100, i, color="#20303d", zorder=3)
        ax.axvline(0, color="#aaa", linestyle="--")
        ax.set(yticks=range(len(names)), yticklabels=[METRICS[n] for n in names],
               title=pair.replace("_minus_", " − ").upper(), xlabel="Score points · 95% CI")
    fig.savefig(output / "deltas.png", dpi=180)
    plt.close(fig)


def save_case_sheets(report, image_sources, output):
    """Keep the fixed example set easy to inspect alongside its question IDs."""
    for group, ids in report["case_selection"]["ids"].items():
        if not ids:
            continue
        sheet = Image.new("RGB", (1160, 400 * ((len(ids) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, qid in enumerate(ids):
            x, y = (index % 2) * 580, (index // 2) * 400
            with Image.open(image_sources[qid]) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((560, 360))
                sheet.paste(preview, (x + (580 - preview.width) // 2, y + 30))
            draw.text((x + 12, y + 8), f"{group} | question {qid}", fill="black")
        sheet.save(output / f"cases_{group}.jpg", quality=90)


def build_study_report(folder, annotations_path, image_root, output_dir, *, study_config,
                       semantic_dir=None):
    folder, output, root = Path(folder), Path(output_dir), Path(image_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = {arm: folder / "validation" / f"{arm}.jsonl" for arm in ARMS}
    inputs["annotations"] = Path(annotations_path)
    inputs["canonical_all"] = Path(annotations_path).parent / "all.jsonl"
    for path in (folder / "training").rglob("*.json"):
        inputs[path.relative_to(folder).as_posix()] = path
    inputs["m1_selection"] = folder / "m1_selection.json"
    inputs.update({f"{arm}_manifest": folder / "validation" / f"{arm}.jsonl.manifest.json"
                   for arm in ARMS})
    inputs["study_config"] = Path(study_config)
    if semantic_dir:
        inputs.update({f"{arm}_semantic": Path(semantic_dir) / f"{arm}_metrics.json"
                       for arm in ARMS})
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    if hashes["annotations"] != VALIDATION_SHA256:
        raise ValueError("Canonical validation changed")
    annotations = list(read_jsonl(annotations_path))
    if len(annotations) != 1407:
        raise ValueError("Require all 1,407 validation questions")
    predictions = {arm: list(read_jsonl(inputs[arm])) for arm in ARMS}
    evidence = verify_evidence(folder, annotations, predictions, study_config,
                               inputs["canonical_all"])
    semantic = ({arm: json.loads(inputs[f"{arm}_semantic"].read_text("utf-8")) for arm in ARMS}
                if semantic_dir else None)
    report = analyze_study(predictions, annotations, semantic=semantic,
                           prediction_hashes={arm: hashes[arm] for arm in ARMS})
    if semantic_dir and any(report["semantic_protocol"].get(k) != v for k, v in {
        "model_id": DEFAULT_BERTSCORE_MODEL, "model_revision": DEFAULT_BERTSCORE_REVISION,
        "num_layers": DEFAULT_BERTSCORE_LAYERS, "idf": False, "rescale_with_baseline": False,
        "use_fast_tokenizer": False, "reference_aggregation": "best_match_across_all_references",
    }.items()):
        raise ValueError("Semantic reports do not use the project's pinned BERTScore protocol")
    image_sources = {}
    for row in predictions["m0"]:
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing image or path outside image root")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['question_id']}")
        image_sources[row["question_id"]] = path
    report["sources"] = {k: {"path": str(p.resolve()), "sha256": hashes[k]}
                         for k, p in inputs.items()}
    report["evidence"] = evidence
    report["images_verified_n"] = len(image_sources)
    report["question_prefix_counts"] = dict(Counter(
        " ".join(r["question"].lower().split()[:2]) for r in annotations).most_common(20))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for qid in {q for ids in report["case_selection"]["ids"].values() for q in ids}:
        if not qid.isdigit():
            raise ValueError("Numeric question IDs required for image filenames")
        with Image.open(image_sources[qid]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((960, 720))
            preview.save(output / "images" / f"{qid}.jpg", quality=90)
    save_study_figures(report, output)
    save_case_sheets(report, image_sources, output)
    (output / "README.md").write_text(render_study_markdown(report), "utf-8")
    (output / "cases.html").write_text(render_study_cases(
        report, {arm: {r["question_id"]: r for r in rows} for arm, rows in predictions.items()},
        {r["question_id"]: r for r in annotations}), "utf-8")
    if any(sha256_file(p) != hashes[k] for k, p in inputs.items()):
        raise ValueError("Inputs changed during report generation")
    write_json(output / "report.json", report)
    write_json(output / "manifest.json", {"files": {
        p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()
    }})
    return report
