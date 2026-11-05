"""Build a local diagnostic bundle from verified validation predictions."""

import csv
import json
import shutil
from html import escape
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.rubric_calibration import check_new_output, verified_validation
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
)
from vqa_explanation.evaluation.validation_diagnostics import (
    METRICS,
    SEEDS,
    TRANSITIONS,
    TYPE_LABELS,
    analyze_validation,
)
from vqa_explanation.inference.study_validation import ARMS
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def render_markdown(report):
    lines = [
        "# Phân tích lỗi validation M0–M3",
        "",
        f"{report['n']:,} câu / {report['n_images']:,} ảnh; ba seed 42, 43, 44. "
        "M0/M1 dùng lại cùng prediction, tổng tám lượt suy luận khác nhau.",
        "",
        "Đây là phân tích thăm dò trên validation, không phải kết quả test hoặc chấm mù.",
        "",
        "## Tổng hợp",
        "",
        "| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, metrics in report["aggregate"].items():
        cells = []
        for value in metrics.values():
            cells.append(
                f"{value['mean'] * 100:.2f}"
                + (f" ± {value['sd'] * 100:.2f}" if value["sd"] is not None else "")
            )
        lines.append(f"| {arm.upper()} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "± là SD qua ba seed, không phải CI. EM đo khớp đáp án tham chiếu; "
        "không thay kiểm đúng/sai theo ảnh.",
        "",
        "## Chuyển đáp án M2 → M3",
        "",
        "| Seed | Cùng khớp | M3 khớp thêm | M3 mất khớp | Cùng không khớp |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        transitions = report["by_seed"][str(seed)]["em_transitions_m2_m3"]
        lines.append(f"| {seed} | " + " | ".join(str(transitions[k]) for k in TRANSITIONS) + " |")
    lines += [
        "",
        "## Nhóm câu hỏi: chênh lệch M3 − M2",
        "",
        "| Nhóm theo quy tắc | Số câu | Δ EM | Δ BERTScore |",
        "|---|---:|---:|---:|",
    ]
    for group in report["groups"]:
        if group["dimension"] != "question_type":
            continue
        delta = group["m3_minus_m2"]
        lines.append(
            f"| {TYPE_LABELS[group['group']]} | {group['n']} | "
            f"{delta['exact_match']['mean'] * 100:+.2f} | "
            f"{delta['bertscore_f1']['mean'] * 100:+.2f} |"
        )
    lines += [
        "",
        "Nhóm được gán từ câu hỏi, không đọc output để chọn nhóm. Đây là nhóm "
        "heuristic, không phải nhãn gốc; xem lại nhóm Khác và các nhóm nhỏ trước khi diễn giải.",
        "Không kiểm định nhiều nhóm hoặc chọn một nhóm thuận lợi làm kết luận chính.",
        "",
        "## Độ ổn định đáp án",
        "",
    ]
    for arm, stats in report["answer_stability"].items():
        p = stats["em_patterns"]
        lines.append(
            f"- {arm.upper()}: {stats['same_normalized_answer_n']}/{report['n']} câu có "
            f"cùng đáp án chuẩn hóa ở cả ba seed; {p['all_match']} câu đều khớp EM, "
            f"{p['mixed']} câu đổi trạng thái khớp, {p['none_match']} câu đều không khớp."
        )
    lines += [
        "",
        "## Sử dụng",
        "",
        "Mở `index.html` để xem toàn bộ câu/ảnh và bốn output, "
        "lọc seed, nhóm hoặc chuyển đáp án. Đây không phải trang chấm mù. "
        "Mọi giải thích/raw output giữ nguyên, không có nhãn hỗ trợ ảnh tự động.",
        "",
        "`groups.csv` chứa chỉ số nhóm, `report.json` chứa số liệu/provenance, "
        "`transitions.svg` là biểu đồ chuyển đáp án. Ảnh trong `images/` là thumbnail "
        "để xem; hash ảnh gốc nằm trong report.",
        "",
        "## Việc còn lại",
        "",
        "1. Hoàn tất luyện rubric và khóa hướng dẫn bằng nhãn chấm thật.",
        "2. Sau khi khóa thiết kế/rubric, xuất test cho tám identity theo kế hoạch hiện hành.",
        "3. Chấm mù 200 ảnh, phân tích hỗ trợ ảnh/chi tiết không được ảnh hỗ trợ.",
        "4. Nghiệm thu giao diện mới với M1 và ảnh ngoài dữ liệu khi có GPU.",
        "Không dùng phân tích validation này để tự tuyên bố DPO tốt hơn hoặc thay giao thức test.",
        "",
    ]
    return "\n".join(lines)


def transition_svg(report):
    colors = {
        "both_match": "#91b7a1",
        "gain": "#497cab",
        "loss": "#ca8a5a",
        "neither_match": "#c8d0d5",
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="280" '
        'viewBox="0 0 1000 280"><rect width="1000" height="280" fill="white"/>',
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#263e48">',
        '<text x="30" y="30" font-size="18">Chuyển khớp đáp án EM: M2 → M3</text>',
    ]
    for i, seed in enumerate(SEEDS):
        y, x = 66 + i * 46, 105
        parts.append(f'<text x="30" y="{y + 20}" font-size="13">Seed {seed}</text>')
        for name in TRANSITIONS:
            count = report["by_seed"][str(seed)]["em_transitions_m2_m3"][name]
            width = count / report["n"] * 850
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="30" '
                f'fill="{colors[name]}"><title>{escape(TRANSITIONS[name])}: {count}</title></rect>'
            )
            if width > 25:
                parts.append(
                    f'<text x="{x + width / 2:.2f}" y="{y + 20}" '
                    f'text-anchor="middle" font-size="11">{count}</text>'
                )
            x += width
    for i, name in enumerate(TRANSITIONS):
        x = 30 + i * 240
        parts.append(f'<rect x="{x}" y="220" width="10" height="10" fill="{colors[name]}"/>')
        parts.append(
            f'<text x="{x + 17}" y="229" font-size="10">{escape(TRANSITIONS[name])}</text>'
        )
    parts.append(
        '<text x="30" y="260" font-size="11">Toàn bộ validation; '
        'EM không phải chấm đúng/sai theo ảnh.</text></g></svg>'
    )
    return "".join(parts)


def build_diagnostics(config_path, environment, output, *, resamples=2000):
    root = project_root()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("schema") != "validation_diagnostics_v1"
        or set(config.get("seeds", {})) != set(SEEDS)
    ):
        raise ValueError("Require the declared three-seed analysis config")

    def resolve(path):
        return (root / path).resolve()

    annotations = environment["annotations"].parent / "validation.jsonl"
    canonical, image_root = environment["annotations"], environment["image_root"].resolve()
    inputs = {"config": config_path, "annotations": annotations, "canonical": canonical}
    data, semantic, hashes, evidence = {}, {}, {}, {}
    protected = [image_root]
    for seed in SEEDS:
        setting = config["seeds"][seed]
        folder, study, semantic_dir = (
            resolve(setting[k]) for k in ("evidence", "study", "semantic")
        )
        protected += [folder, study, semantic_dir]
        paths, data[seed], evidence[seed] = verified_validation(
            folder, annotations, canonical, study
        )
        if (
            evidence[seed]["training_seed"] != seed
            or evidence[seed]["training_learning_rate"] != 1e-5
        ):
            raise ValueError("Diagnostics config mixed training seeds or LR sensitivity runs")
        hashes[seed] = {a: sha256_file(p) for a, p in paths.items()}
        semantic[seed] = {}
        inputs[f"study_{seed}"] = study
        inputs[f"selection_{seed}"] = folder / "m1_selection.json"
        for p in (folder / "training").rglob("*.json"):
            inputs[f"training_{seed}/{p.relative_to(folder).as_posix()}"] = p
        for arm in ARMS:
            inputs[f"{seed}_{arm}"] = paths[arm]
            inputs[f"{seed}_{arm}_manifest"] = paths[arm].with_suffix(".jsonl.manifest.json")
            metric_path = semantic_dir / f"{arm}_metrics.json"
            inputs[f"{seed}_{arm}_semantic"] = metric_path
            semantic[seed][arm] = json.loads(metric_path.read_text("utf-8"))
        print(f"Verified validation evidence: seed {seed}", flush=True)
    output = check_new_output(output, (*inputs.values(), *protected))
    input_hashes = {k: sha256_file(p) for k, p in inputs.items()}
    report = analyze_validation(
        data, list(read_jsonl(annotations)), semantic, hashes, resamples=resamples
    )
    expected_semantic = {
        "model_id": DEFAULT_BERTSCORE_MODEL,
        "model_revision": DEFAULT_BERTSCORE_REVISION,
        "num_layers": DEFAULT_BERTSCORE_LAYERS,
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "reference_aggregation": "best_match_across_all_references",
    }
    for seed in SEEDS:
        protocol = semantic[seed]["m0"]["explanation_bertscore"]["protocol"]
        if any(protocol.get(key) != value for key, value in expected_semantic.items()):
            raise ValueError("Semantic reports do not use the pinned project protocol")
    images = {}
    for case in report["cases"]:
        path = (image_root / case["image_path"]).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError("Image is missing or outside image root")
        if path not in images:
            digest = sha256_file(path)
            if digest != case["image_sha256"]:
                raise ValueError("Image checksum differs from exported prediction")
            images[path] = digest
        elif images[path] != case["image_sha256"]:
            raise ValueError("Shared image has inconsistent hashes")
        case["thumbnail"] = f"images/{case['image_sha256']}.jpg"
    report["sources"] = {k: {"path": str(p), "sha256": input_hashes[k]} for k, p in inputs.items()}
    report["images_verified"] = len(images)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for path, digest in images.items():
        with Image.open(path) as original:
            thumbnail = ImageOps.exif_transpose(original).convert("RGB")
            thumbnail.thumbnail((640, 480))
            thumbnail.save(output / "images" / f"{digest}.jpg", quality=82)
    templates = root / "app/validation_explorer"
    for source, dest in (
        ("validation_explorer.html", "index.html"),
        ("validation_explorer.css", "explorer.css"),
        ("validation_explorer.js", "explorer.js"),
    ):
        shutil.copyfile(templates / source, output / dest)
    payload = {k: report[k] for k in ("n", "n_images", "type_labels", "transition_labels", "cases")}
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output / "data.js").write_text("const VALIDATION_DATA = " + encoded + ";\n", "utf-8")
    (output / "README.md").write_text(render_markdown(report), "utf-8")
    (output / "transitions.svg").write_text(transition_svg(report), "utf-8")
    with (output / "groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "group", "n", "n_images", "seed", "arm", *METRICS])
        for group in report["groups"]:
            for seed in SEEDS:
                for arm in ARMS:
                    writer.writerow(
                        [
                            group["dimension"],
                            group["group"],
                            group["n"],
                            group["n_images"],
                            seed,
                            arm,
                            *(group["scores"][str(seed)][arm][m] for m in METRICS),
                        ]
                    )
    if any(sha256_file(p) != input_hashes[k] for k, p in inputs.items()):
        raise ValueError("Evidence changed while generating diagnostics")
    if any(sha256_file(p) != digest for p, digest in images.items()):
        raise ValueError("Original images changed while generating diagnostics")
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "schema": "validation_diagnostics_bundle_v1",
            "files": {
                p.relative_to(output).as_posix(): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    return report
