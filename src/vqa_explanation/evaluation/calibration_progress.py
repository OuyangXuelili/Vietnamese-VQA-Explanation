"""Check calibration drafts without completing ratings or freezing a rubric."""

import csv
import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.blind_review import summarize_blind_review
from vqa_explanation.evaluation.review_progress import inspect_review_progress
from vqa_explanation.evaluation.rubric_calibration import check_new_output
from vqa_explanation.utils.run_provenance import sha256_file

RATING_FIELDS = ("answer_assessment", "unsupported_detail", "explanation_support")


def inspect_calibration(calibration, rating_paths=None):
    calibration = Path(calibration).resolve()
    manifest_path = calibration / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("schema") != "rubric_calibration_v1"
            or manifest.get("split") != "validation" or manifest.get("images") != 30
            or manifest.get("outputs_per_rater") != 120):
        raise ValueError("Require the 30-image validation calibration")
    for name, expected in manifest["files"].items():
        path = (calibration / name).resolve()
        if not path.is_relative_to(calibration) or sha256_file(path) != expected:
            raise ValueError("Calibration bundle changed")
    bundle = calibration / "bundle"
    private = json.loads((bundle / "private/manifest.json").read_text("utf-8"))
    if private.get("split") != "validation" or private.get("sample_count") != 30:
        raise ValueError("Require validation-only calibration pages")
    inspection = inspect_review_progress(bundle, rating_paths)
    ratings = {key: Path(inspection["sources"]["rating:" + key]["path"])
               for key in inspection["raters"]}
    sources = {"calibration_manifest": manifest_path,
               **{key: Path(value["path"]) for key, value in inspection["sources"].items()}}
    hashes = {"calibration_manifest": manifest_hash,
              **{key: value["sha256"] for key, value in inspection["sources"].items()}}
    progress = inspection["raters"]
    indexed = {rater: {r["entry_id"]: r for r in read_jsonl(path)}
               for rater, path in ratings.items()}
    complete = all(p["remaining"] == 0 for p in progress.values())
    summary, disagreements = None, None
    if complete:
        summary = summarize_blind_review(bundle, ratings)
        if len(ratings) == 2:
            key = list(read_jsonl(bundle / "private/model_key.jsonl"))
            paired = {}
            for mapping in key:
                if mapping["rater_id"] in indexed:
                    pair = (mapping["question_id"], mapping["experiment_id"])
                    paired.setdefault(pair, {})[mapping["rater_id"]] = indexed[
                        mapping["rater_id"]][mapping["entry_id"]]
            disagreements = []
            for (qid, experiment), rows in sorted(paired.items()):
                if set(rows) != set(ratings):
                    raise ValueError("Calibration key does not pair the independent ratings")
                for field in RATING_FIELDS:
                    values = {rater: row[field] for rater, row in rows.items()}
                    if len(set(values.values())) > 1:
                        disagreements.append({"question_id": qid, "experiment_id": experiment,
                                              "field": field, "ratings": values})
    if any(sha256_file(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("Calibration or ratings changed during inspection")
    return {"schema": "calibration_progress_v1", "split": "validation",
            "calibration": str(calibration),
            "status": "ready_for_discussion" if complete and len(ratings) == 2
            else "one_rater_complete" if complete else "awaiting_ratings",
            "images": 30, "raters": progress, "summary": summary,
            "disagreements": disagreements,
            "sources": {key: {"path": str(path), "sha256": hashes[key]}
                        for key, path in sources.items()},
            "rubric_frozen": False, "test_predictions_generated": False,
            "limitation": "Private research report. Do not show other ratings before independent "
            "completion. Metadata does not prove human authorship or independence."}


def save_calibration_progress(report, output):
    sources = report["sources"].values()
    if any(sha256_file(Path(item["path"])) != item["sha256"] for item in sources):
        raise ValueError("Calibration or ratings changed after inspection")
    output = check_new_output(output, [report["calibration"],
                                      *(item["path"] for item in sources)])
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    lines = ["# Tiến độ chấm thử", "", "30 ảnh validation, bốn đầu ra mỗi ảnh.", "",
             "| File nhãn | Đủ trường | Còn lại |", "|---|---:|---:|"]
    for rater, counts in report["raters"].items():
        lines.append(f"| {rater} | {counts['complete']}/{counts['expected']} "
                     f"| {counts['remaining']} |")
    if report["status"] == "awaiting_ratings":
        lines += ["", "Chưa đủ nhãn. report.json ghi từng mục và trường còn thiếu.",
                  "Chấm độc lập trên trang của từng người, xuất JSONL rồi kiểm lại.",
                  "Chưa tính điểm, độ đồng thuận hoặc mở nhãn của người còn lại."]
    else:
        lines += ["", "Các file đã đủ trường. Đọc summary trong report.json; thảo luận bất đồng",
                  "sau khi lưu nhãn độc lập. Một người chấm thì không có độ đồng thuận."]
    lines += ["", "Báo cáo riêng cho phía nghiên cứu. Không tự sửa nhãn hoặc khóa rubric.",
              "Bộ chấm thử không phải kết quả test. Nhãn gốc được giữ nguyên.", ""]
    (output / "README.md").write_text("\n".join(lines), "utf-8")
    if report["disagreements"] is not None:
        with (output / "disagreements.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            raters = list(report["raters"])
            writer.writerow(["question_id", "experiment_id", "field", *raters])
            for row in report["disagreements"]:
                writer.writerow([row["question_id"], row["experiment_id"], row["field"],
                                 *(row["ratings"][rater] for rater in raters)])
    write_json(output / "manifest.json", {"files": {
        path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}})
    return output
