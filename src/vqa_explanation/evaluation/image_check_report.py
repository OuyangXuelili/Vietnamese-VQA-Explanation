"""Verify and compare the three completed image-permutation arms."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation.answer_metrics import exact_match, normalize_answer
from vqa_explanation.evaluation.comparison import cluster_bootstrap
from vqa_explanation.evaluation.image_dependence import score_image_check, verify_images
from vqa_explanation.evaluation.study_report import verify_evidence
from vqa_explanation.inference.export import select_records
from vqa_explanation.inference.study_validation import ARMS
from vqa_explanation.utils.run_provenance import sha256_file

IMAGE_ARMS = ("m1", "m2", "m3")


def paired_image_results(predictions, original, *, seed=42, resamples=2000):
    """Pair both conditions and all arms by question, never by file order."""
    if set(predictions) != set(IMAGE_ARMS):
        raise ValueError("Require M1, M2 and M3")
    canonical = {row["question_id"]: row for row in original}
    if (not canonical or len(canonical) != len(original)
            or len({r["image_id"] for r in original}) != len(original)
            or any(r["split"] != "validation" for r in original)):
        raise ValueError("Require distinct validation images")
    ids = sorted(canonical)
    values, items = {}, []
    for arm in IMAGE_ARMS:
        indexed = {}
        if set(predictions[arm]) != {"correct", "permuted"}:
            raise ValueError("Require both image conditions")
        for condition, rows in predictions[arm].items():
            indexed[condition] = {r["question_id"]: r for r in rows}
            if len(rows) != len(ids) or set(indexed[condition]) != set(ids):
                raise ValueError("Incomplete or duplicate paired predictions")
        values[arm] = {key: [] for key in (
            "correct_image_em", "permuted_image_em_against_original_answer", "em_drop",
            "answer_change_rate", "explanation_change_rate")}
        for qid in ids:
            left, right = (indexed[c][qid] for c in ("correct", "permuted"))
            correct = exact_match(left["prediction_answer"], canonical[qid]["answer"])
            permuted = exact_match(right["prediction_answer"], canonical[qid]["answer"])
            scores = {
                "correct_image_em": correct,
                "permuted_image_em_against_original_answer": permuted,
                "em_drop": correct - permuted,
                "answer_change_rate": int(normalize_answer(left["prediction_answer"]) !=
                                          normalize_answer(right["prediction_answer"])),
                "explanation_change_rate": int(
                    " ".join(left["prediction_explanation"].casefold().split()) !=
                    " ".join(right["prediction_explanation"].casefold().split())),
            }
            for key, value in scores.items():
                values[arm][key].append(value)
            items.append({"arm": arm, "question_id": qid,
                          "image_id": canonical[qid]["image_id"], **scores})
    comparisons = {}
    for left, right in combinations(IMAGE_ARMS, 2):
        differences = {key: [b - a for a, b in zip(values[left][key], values[right][key],
                                                  strict=True)] for key in values[left]}
        comparisons[f"{right}_minus_{left}"] = cluster_bootstrap(
            differences, [canonical[q]["image_id"] for q in ids],
            seed=seed, resamples=resamples)
    return {
        "schema": "three_arm_image_permutation_report_v1", "split": "validation",
        "questions": len(ids), "bootstrap_seed": seed, "bootstrap_resamples": resamples,
        "scores": {arm: {key: sum(v) / len(v) for key, v in scores.items()}
                   for arm, scores in values.items()},
        "comparisons": comparisons,
        "limitations": [
            "One fixed permutation and one question per image; not a full-validation score.",
            "Permuted-image EM uses the original answer, not a label for the donor image.",
            "Changed answers or explanations measure sensitivity, not visual correctness.",
            "The donor permutation couples observations; bootstrap CIs are descriptive, "
            "conditional on this mapping, not independent experimental replications.",
            "One training seed; do not infer DPO superiority or equivalence from this check.",
        ],
    }, items


def build_image_report(results, evidence, annotations, image_root, study_config, output_dir):
    results, evidence, output = Path(results), Path(evidence), Path(output_dir)
    if output.exists():
        raise FileExistsError("Preserve the previous image diagnostic report")
    bundle = results / "input"
    diagnostic, data = verify_images(bundle, annotations, image_root)
    validation = [r for r in read_jsonl(annotations) if r["split"] == "validation"]
    full_predictions = {arm: list(read_jsonl(evidence / "validation" / f"{arm}.jsonl"))
                        for arm in ARMS}
    verified = verify_evidence(evidence, validation, full_predictions, study_config, annotations)
    snapshot = json.loads((results / "snapshot.json").read_text("utf-8"))
    expected_names = {f"input/{name}" for name in data} | {"input/manifest.json"}
    expected_names |= {f"predictions/{arm}/{name}" for arm in IMAGE_ARMS for name in (
        "correct_images.jsonl", "correct_images.jsonl.manifest.json",
        "permuted_images.jsonl", "permuted_images.jsonl.manifest.json", "image_dependence.json")}
    if (snapshot.get("schema") != "image_permutation_snapshot_v1"
            or set(snapshot.get("files", {})) != expected_names
            or any(sha256_file(results / name) != digest
                   for name, digest in snapshot["files"].items())):
        raise ValueError("Image snapshot files changed")
    predictions, recomputed, agreements = {}, {}, {}
    identity_fields = ("model_id", "model_revision", "experiment_id", "adapter_files",
                       "split", "limit", "seed", "min_pixels", "max_pixels", "max_new_tokens",
                       "load_in_4bit", "source_files", "python", "versions")
    driver_hash = sha256_file(Path(__file__).with_name("image_dependence.py"))
    for arm in IMAGE_ARMS:
        directory = results / "predictions" / arm
        recomputed[arm] = score_image_check(bundle, annotations, image_root, directory)
        if json.loads((directory / "image_dependence.json").read_text("utf-8")) != recomputed[arm]:
            raise ValueError("Saved image score differs from predictions")
        baseline = verified["manifests"][arm]
        predictions[arm] = {}
        for condition, name in (("correct", "correct_images.jsonl"),
                                ("permuted", "permuted_images.jsonl")):
            manifest = json.loads((directory / f"{name}.manifest.json").read_text("utf-8"))
            inputs = manifest["inputs"]
            selected = select_records(data[name], "validation", 0, diagnostic["seed"])
            selected_hash = hashlib.sha256(json.dumps(
                selected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if (any(inputs.get(k) != baseline["inputs"].get(k) for k in identity_fields)
                    or manifest["protocol"] != baseline["protocol"]
                    or inputs.get("selected_sha256") != selected_hash
                    or inputs.get("context", {}).get("driver_sha256") != driver_hash):
                raise ValueError(f"{arm}: diagnostic differs from verified inference identity")
            predictions[arm][condition] = list(read_jsonl(directory / name))
        full = {r["question_id"]: r for r in full_predictions[arm]}
        agreements[arm] = {
            f"{field}_exact_string_n": sum(
                r[f"prediction_{field}"] == full[r["question_id"]][f"prediction_{field}"]
                for r in predictions[arm]["correct"])
            for field in ("answer", "explanation")}
    report, items = paired_image_results(predictions, data["correct_images.jsonl"])
    report.update({"single_arm_recomputed": recomputed, "full_validation_agreement": agreements,
                   "input_evidence": snapshot,
                   "bundle_sha256": sha256_file(bundle / "manifest.json"),
                   "source_sha256": sha256_file(Path(__file__)),
                   "study_sha256": sha256_file(Path(study_config)),
                   "training_reports": {
                       a: sha256_file(evidence / "training" / a / "training_result.json")
                       for a in ("m2", "m3")}})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    write_jsonl(output / "items.jsonl", items)
    lines = ["# Đối chứng đổi ảnh — seed 42", "",
             "200 câu validation, mỗi ảnh một câu. Cùng câu hỏi và cấu hình; "
             "chỉ hoán đổi ảnh. Đủ 1.200 dự đoán, đã kiểm raw output, checkpoint và manifest.", "",
             "| Nhánh | EM ảnh đúng | EM ảnh đổi | Giảm EM | Đổi đáp án |",
             "|---|---:|---:|---:|---:|"]
    for arm, scores in report["scores"].items():
        lines.append(f"| {arm.upper()} | " + " | ".join(f"{scores[k] * 100:.1f}" for k in (
            "correct_image_em", "permuted_image_em_against_original_answer", "em_drop",
            "answer_change_rate")) + " |")
    lines += ["", "Đơn vị: %. EM ảnh đổi vẫn đối chiếu đáp án của ảnh gốc, "
              "không phải độ chính xác trên ảnh thay thế.", "", "## So sánh M3 − M2", ""]
    for key, label in (("em_drop", "Mức giảm EM"), ("answer_change_rate", "Tỷ lệ đổi đáp án")):
        metric = report["comparisons"]["m3_minus_m2"]["metrics"][key]
        lo, hi = metric["ci95"]
        lines.append(f"{label}: {metric['delta_right_minus_left'] * 100:+.1f} điểm %, "
                     f"CI 95% [{lo * 100:+.1f}; {hi * 100:+.1f}].")
    lines += ["", "Cả ba nhánh phụ thuộc rõ vào ảnh ở phép kiểm này. "
              "Chưa có bằng chứng M3 tốt hơn M2 về độ nhạy với ảnh. "
              "Không dùng kết quả này để kết luận lời giải thích đúng theo ảnh.", "",
              "CI bootstrap 2.000 lần ghép theo ảnh gốc. Đây là mô tả có điều kiện "
              "trên một phép hoán đổi; mỗi ảnh còn được dùng làm ảnh thay thế nên "
              "các quan sát không hoàn toàn độc lập. Chưa đo độ ổn định qua seed.", "",
              "Bước tiếp: lặp M2/M3 với seed 43 và 44, giữ nguyên dữ liệu và ngân sách. "
              "Test vẫn đóng; chấm mù theo ảnh thực hiện sau khi khóa thiết kế.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "manifest.json", {p.name: sha256_file(p) for p in output.iterdir()})
    return report
