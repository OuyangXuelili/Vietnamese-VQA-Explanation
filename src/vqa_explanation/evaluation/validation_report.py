"""Build a reproducible validation report from saved M0/M1 predictions."""

import hashlib
import json
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean

from PIL import Image, ImageOps

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.answer_metrics import exact_match, token_f1
from vqa_explanation.evaluation.comparison import (
    PROTOCOL_FIELDS,
    cluster_bootstrap,
    compare_predictions,
)
from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1
from vqa_explanation.evaluation.semantic_metrics import compare_bertscore_reports
from vqa_explanation.utils.run_provenance import sha256_file

GROUPS = {
    "both_match": "Cả hai khớp tham chiếu",
    "gain": "M0 không khớp → M1 khớp",
    "loss": "M0 khớp → M1 không khớp",
    "neither_match": "Cả hai không khớp tham chiếu",
}
METRICS = {"exact_match": "EM", "token_f1": "Token-F1", "rouge_l": "ROUGE-L",
           "bertscore_f1": "BERTScore F1"}


def analyze_predictions(left, right, annotations, *, seed=42, resamples=2000, per_group=6):
    if type(per_group) is not int or per_group < 1:
        raise ValueError("Cases per group must be a positive integer")
    canonical = {}
    for row in annotations:
        if row["question_id"] in canonical or row["split"] != "validation":
            raise ValueError("Report requires unique validation annotations")
        canonical[row["question_id"]] = row
    comparison = compare_predictions(left, right, expected_annotations=canonical)
    if not comparison["comparable"]:
        raise ValueError(f"Predictions are not comparable: {comparison['issues'][:5]}")
    if not comparison["explanation_rouge_l_f1"]["comparable"]:
        raise ValueError("Explanation references or predictions are invalid")
    indexed = [{r["question_id"]: r for r in rows} for rows in (left, right)]
    ids = sorted(canonical)
    values, diagnostics = {}, {}
    for label, data in zip(("M0", "M1"), indexed, strict=True):
        rows = [data[q] for q in ids]
        values[label] = {
            "exact_match": [exact_match(r["prediction_answer"], r["reference_answer"])
                            for r in rows],
            "token_f1": [token_f1(r["prediction_answer"], r["reference_answer"]) for r in rows],
            "rouge_l": [rouge_l_f1(r["prediction_explanation"], r["reference_explanations"])
                        for r in rows],
        }
        if any(type(r.get("format_valid")) is not bool for r in rows):
            raise ValueError("Predictions must include exporter format_valid flags")
        diagnostics[label] = {
            "format_valid_n": sum(r["format_valid"] for r in rows),
            "empty_answer_n": sum(not r["prediction_answer"].strip() for r in rows),
            "empty_explanation_n": sum(not r["prediction_explanation"].strip() for r in rows),
            "mean_explanation_whitespace_tokens": mean(
                len(r["prediction_explanation"].split()) for r in rows),
            "distinct_explanations": len({r["prediction_explanation"].strip() for r in rows}),
        }
    groups = {group: [] for group in GROUPS}
    for i, qid in enumerate(ids):
        a, b = values["M0"]["exact_match"][i], values["M1"]["exact_match"][i]
        group = "both_match" if a and b else "loss" if a else "gain" if b else "neither_match"
        groups[group].append(qid)
    selected = {}
    for group, qids in groups.items():
        ordered = sorted(qids, key=lambda q: (
            hashlib.sha256(f"validation_case_v1:{seed}:{group}:{q}".encode()).hexdigest(), q))
        seen, selected[group] = set(), []
        for qid in ordered:
            image_id = canonical[qid]["image_id"]
            if image_id not in seen:
                selected[group].append(qid)
                seen.add(image_id)
            if len(selected[group]) == per_group:
                break
    differences = {name: [b - a for a, b in zip(values["M0"][name], values["M1"][name],
                                               strict=True)] for name in values["M0"]}
    return {
        "schema": "m0_m1_validation_report_v1", "split": "validation", "n": len(ids),
        "n_images": len({r["image_id"] for r in annotations}),
        "comparison": comparison,
        "scores": {label: {name: mean(v) for name, v in data.items()}
                   for label, data in values.items()},
        "bootstrap": cluster_bootstrap(differences, [canonical[q]["image_id"] for q in ids],
                                       seed=seed, resamples=resamples),
        "diagnostics": diagnostics, "em_transitions": {k: len(v) for k, v in groups.items()},
        "case_selection": {"seed": seed, "per_group": per_group, "ids": selected,
                           "method": "SHA-256 rank within EM transition; one question per image "
                                     "within each group; no visual judgement"},
        "inference_protocol": {key: left[0][key] for key in PROTOCOL_FIELDS},
        "adapters": {label: rows[0]["adapter"] for label, rows in (("M0", left), ("M1", right))},
    }


def render_markdown(report):
    rows = ["# Kết quả M0 và M1 trên validation", "",
            f"Phân tích {report['n']:,} câu trên {report['n_images']:,} ảnh. "
            "M1 là mô hình đã chọn bằng validation; các số dưới đây chưa phải kết quả test.", "",
            "| Chỉ số (×100) | M0 | M1 | M1 − M0 | CI 95% của chênh lệch |",
            "|---|---:|---:|---:|---:|"]
    for name in report["scores"]["M0"]:
        a, b = (report["scores"][label][name] * 100 for label in ("M0", "M1"))
        lo, hi = (x * 100 for x in report["bootstrap"]["metrics"][name]["ci95"])
        rows.append(f"| {METRICS[name]} | {a:.2f} | {b:.2f} | {b-a:+.2f} | [{lo:.2f}; {hi:.2f}] |")
    rows += ["", "![Điểm và khoảng tin cậy](metrics.png)", "",
             f"CI dùng paired bootstrap theo cụm ảnh, {report['bootstrap']['resamples']:,} lần, "
             f"seed {report['bootstrap']['seed']}. "
             "Nó phản ánh bất định của tập validation trong hai run này; "
             "chưa đo độ ổn định giữa các seed train.", "",
             "## Đáp án thay đổi thế nào", "",
             "| Nhóm theo exact match | Số câu |", "|---|---:|"]
    rows += [f"| {GROUPS[k]} | {n} |" for k, n in report["em_transitions"].items()]
    rows += ["", "![Chuyển trạng thái exact match](answer_transitions.png)", "",
             "Không khớp tham chiếu có thể do từ đồng nghĩa hoặc annotation chưa chính xác. "
             "Các nhóm này chưa phải phân loại đúng/sai bằng ảnh.", "",
             "## Định dạng và độ dài", "",
             "| Mô hình | Đúng định dạng | Đáp án rỗng | Giải thích rỗng | "
             "Độ dài giải thích trung bình |", "|---|---:|---:|---:|---:|"]
    for label, d in report["diagnostics"].items():
        rows.append(f"| {label} | {d['format_valid_n']}/{report['n']} | "
                    f"{d['empty_answer_n']} | {d['empty_explanation_n']} | "
                    f"{d['mean_explanation_whitespace_tokens']:.2f} |")
    rows += ["", "Độ dài tính bằng tách khoảng trắng; đây không phải tokenizer của Qwen. "
             "Cờ định dạng đọc từ prediction exporter.", "",
             "## Đọc ví dụ và diễn giải", "",
             "Mở [cases.html](cases.html) để xem ảnh, đầu ra hai nhánh và tham chiếu. "
             "Ví dụ được chọn bằng quy tắc cố định trong từng nhóm EM, "
             "không phải các trường hợp được người chấm chọn là tốt nhất. "
             "Các ảnh ví dụ không dùng để ước lượng tỷ lệ lỗi toàn tập.", "",
             "M1 tăng điểm tự động trên cùng validation. ROUGE-L và BERTScore đo mức "
             "giống văn bản tham chiếu; chưa đủ kết luận giải thích được ảnh hỗ trợ. "
             "Phần kiểm người bằng ảnh và đối chứng M2–M3 vẫn cần thực hiện.", "",
             "Nguồn, hash file đầu vào, giao thức suy luận và số liệu đầy đủ nằm trong "
             "`report.json`. Không có suy luận mới hoặc thay đổi checkpoint trong bước này."]
    return "\n".join(rows) + "\n"


def render_cases(report, left, right, canonical):
    pages = ["<!doctype html><html lang='vi'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Phân tích validation M0/M1</title><style>",
             "body{font:16px/1.5 Arial;max-width:1100px;margin:auto;padding:24px;color:#20303d}",
             "article{border-top:1px solid #ccc;padding:24px 0}img{max-width:100%;height:280px;"
             "object-fit:contain}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:24px}",
             "p{white-space:pre-wrap}.muted{color:#5b6470}@media(max-width:700px){"
             ".outputs{grid-template-columns:1fr}}@media print{article{break-inside:avoid}}",
             "</style><h1>M0/M1: ví dụ validation</h1><p>Ví dụ theo nhóm exact match. "
             "Chưa có kết luận kiểm ảnh. Tham chiếu có thể có lỗi; "
             "khác từ không nhất thiết sai.</p>"]
    for group, ids in report["case_selection"]["ids"].items():
        pages.append(f"<h2>{GROUPS[group]}</h2>")
        for qid in ids:
            ref = canonical[qid]
            pages.append(f"<article><h3>{escape(ref['question'])}</h3>"
                         f"<p class='muted'>Câu {escape(qid)} · ảnh {escape(ref['image_id'])}</p>"
                         f"<img src='images/{escape(qid, quote=True)}.jpg' alt='Ảnh câu hỏi'>"
                         "<div class='outputs'>")
            for label, data in (("M0", left), ("M1", right)):
                row = data[qid]
                pages.append(f"<section><h4>{label}</h4><p><b>Đáp án:</b> "
                             f"{escape(row['prediction_answer'])}</p><p><b>Giải thích:</b> "
                             f"{escape(row['prediction_explanation'])}</p></section>")
            refs = "".join(f"<li>{escape(s)}</li>" for s in ref["explanations"])
            pages.append(f"</div><details><summary>Tham chiếu: {escape(ref['answer'])}</summary>"
                         f"<ul>{refs}</ul></details></article>")
    return "\n".join(pages) + "</html>"


def save_figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(report["scores"]["M0"])
    figure, (ax, delta_ax) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    x = np.arange(len(names))
    for offset, label, color in ((-.19, "M0", "#66758c"), (.19, "M1", "#087f8c")):
        bars = ax.bar(x + offset, [report["scores"][label][n] * 100 for n in names],
                      .36, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set(xticks=x, xticklabels=[METRICS[n] for n in names], ylim=(0, 100),
           ylabel="Score × 100", title=f"Validation: {report['n']} questions")
    ax.legend(frameon=False)
    for i, name in enumerate(names):
        stat = report["bootstrap"]["metrics"][name]
        lo, hi = [v * 100 for v in stat["ci95"]]
        delta = stat["delta_right_minus_left"] * 100
        delta_ax.plot([lo, hi], [i, i], color="#087f8c", linewidth=3)
        delta_ax.scatter([delta], [i], color="#20303d", zorder=3)
    delta_ax.axvline(0, color="#aaa", linestyle="--")
    delta_ax.set(yticks=x, yticklabels=[METRICS[n] for n in names],
                 xlabel="M1 − M0 (score points)", title="Paired image bootstrap: 95% CI")
    figure.savefig(output / "metrics.png", dpi=180)
    plt.close(figure)
    counts = report["em_transitions"]
    matrix = np.array([[counts["neither_match"], counts["gain"]],
                       [counts["loss"], counts["both_match"]]])
    figure, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=20,
                color="white" if value > matrix.max() / 2 else "#20303d")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No match", "Match"],
           yticklabels=["No match", "Match"], xlabel="M1 exact match", ylabel="M0 exact match",
           title="Answer transitions against one reference")
    figure.savefig(output / "answer_transitions.png", dpi=180)
    plt.close(figure)


def build_validation_report(left_path, right_path, annotations_path, image_root, output_dir,
                            *, annotations_sha256, semantic_paths=None):
    inputs = {"m0_predictions": Path(left_path), "m1_predictions": Path(right_path),
              "annotations": Path(annotations_path)}
    if semantic_paths:
        if len(semantic_paths) != 2:
            raise ValueError("Both M0/M1 semantic reports are required")
        inputs.update(zip(("m0_semantic", "m1_semantic"), map(Path, semantic_paths), strict=True))
    hashes = {k: sha256_file(p) for k, p in inputs.items()}
    if hashes["annotations"] != annotations_sha256:
        raise ValueError("Canonical validation hash mismatch")
    output, root = Path(output_dir), Path(image_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    left, right, annotations = (list(read_jsonl(inputs[k]))
                                for k in ("m0_predictions", "m1_predictions", "annotations"))
    if len(annotations) != 1407:
        raise ValueError("Official validation report requires all 1,407 annotations")
    report = analyze_predictions(left, right, annotations)
    if semantic_paths:
        semantic, differences, images = compare_bertscore_reports(
            *(json.loads(inputs[k].read_text("utf-8")) for k in ("m0_semantic", "m1_semantic")),
            left_predictions_sha256=hashes["m0_predictions"],
            right_predictions_sha256=hashes["m1_predictions"], expected_split="validation",
            expected_question_images={r["question_id"]: r["image_id"] for r in annotations})
        for label, side in (("M0", "left"), ("M1", "right")):
            report["scores"][label]["bertscore_f1"] = semantic["scores"][side]["f1"]
        report["bootstrap"]["metrics"].update(
            cluster_bootstrap({"bertscore_f1": differences}, images)["metrics"])
        report["semantic_protocol"] = semantic["protocol"]
    canonical = {r["question_id"]: r for r in annotations}
    a, b = ({r["question_id"]: r for r in records} for records in (left, right))
    image_sources = {}
    for row in left:
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing validation image or path outside image root")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError(f"Image changed since inference: {row['question_id']}")
        image_sources[row["question_id"]] = path
    report["sources"] = {k: {"path": str(p.resolve()), "sha256": hashes[k]}
                         for k, p in inputs.items()}
    report["images_verified_n"] = len(image_sources)
    report["question_prefix_counts"] = dict(Counter(
        " ".join(r["question"].lower().split()[:2]) for r in annotations).most_common(20))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for qid in {q for ids in report["case_selection"]["ids"].values() for q in ids}:
        if not qid.isdigit():
            raise ValueError("Canonical question ID must be numeric for preview filenames")
        with Image.open(image_sources[qid]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((960, 720))
            preview.save(output / "images" / f"{qid}.jpg", quality=90)
    save_figures(report, output)
    (output / "README.md").write_text(render_markdown(report), "utf-8")
    (output / "cases.html").write_text(render_cases(report, a, b, canonical), "utf-8")
    if any(sha256_file(p) != hashes[k] for k, p in inputs.items()):
        raise ValueError("Report inputs changed during analysis")
    write_json(output / "report.json", report)
    write_json(output / "manifest.json", {"files": {
        p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()
    }})
    return report
