"""Validate saved rating drafts without scoring incomplete work."""

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.blind_review import LABEL_FIELDS, render_blind_editor
from vqa_explanation.utils.review_time import parse_review_time
from vqa_explanation.utils.run_provenance import sha256_file

CHOICES = {
    "answer_assessment": {"correct", "incorrect", "uncertain"},
    "unsupported_detail": {"yes", "no", "uncertain"},
}
TEXT_FIELDS = ("reviewer", "reviewed_at", "image_evidence")


def missing_fields(row):
    if row.get("review_status") not in {"pending", "reviewed"}:
        raise ValueError("Invalid review status")
    missing = []
    for field, choices in CHOICES.items():
        value = row.get(field)
        if value in (None, ""):
            missing.append(field)
        elif not isinstance(value, str) or value not in choices:
            raise ValueError(f"Invalid label: {field}")
    support = row.get("explanation_support")
    if support in (None, ""):
        missing.append("explanation_support")
    elif type(support) is not int or support not in {0, 1, 2}:
        raise ValueError("Invalid label: explanation_support")
    for field in TEXT_FIELDS:
        value = row.get(field)
        if value is None or isinstance(value, str) and not value.strip():
            missing.append(field)
        elif not isinstance(value, str):
            raise ValueError(f"Invalid text field: {field}")
    if "reviewed_at" not in missing:
        parse_review_time(row["reviewed_at"])
    if row["review_status"] != "reviewed":
        missing.append("review_status")
    return sorted(missing)


def inspect_review_progress(bundle, rating_paths=None):
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "private/manifest.json"
    hashes = {"bundle_manifest": sha256_file(manifest_path)}
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("schema") != "blind_review_bundle_v1"
            or manifest.get("split") not in {"validation", "test"}
            or type(manifest.get("sample_count")) is not int
            or manifest["sample_count"] < 2):
        raise ValueError("Invalid blind review manifest")
    sources = {"bundle_manifest": manifest_path}
    required = {"private/model_key.jsonl"} | {
        f"{rater}/{name}" for rater in manifest["raters"]
        for name in ("index.html", "blank_ratings.jsonl")}
    if not required.issubset(manifest["files"]):
        raise ValueError("Review manifest does not cover the blinded pages and sources")
    for relative, digest in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or sha256_file(path) != digest:
            raise ValueError("Blind review bundle changed")
        sources["bundle:" + relative] = path
        hashes["bundle:" + relative] = digest
    ratings = {key: Path(path).resolve() for key, path in (rating_paths or {
        key: bundle / key / "blank_ratings.jsonl" for key in manifest["raters"]
    }).items()}
    if not ratings or not set(ratings).issubset(manifest["raters"]):
        raise ValueError("Unknown or empty rater set")
    for key, path in ratings.items():
        sources["rating:" + key] = path
        hashes["rating:" + key] = sha256_file(path)
    progress, identities = {}, set()
    for rater, path in ratings.items():
        baseline = {r["entry_id"]: r for r in read_jsonl(bundle / rater / "blank_ratings.jsonl")}
        rows = list(read_jsonl(path))
        if (len(baseline) != 4 * manifest["sample_count"] or len(rows) != len(baseline)
                or {r["entry_id"] for r in rows} != set(baseline)):
            raise ValueError("Ratings must cover every blinded output exactly once")
        pending, reviewers, images = [], set(), defaultdict(list)
        for row in rows:
            original = baseline[row["entry_id"]]
            if any(row.get(k) != v for k, v in original.items() if k not in LABEL_FIELDS):
                raise ValueError("Immutable blinded output changed")
            missing = missing_fields(row)
            images[row["image_path"]].append(not missing)
            if missing:
                pending.append({"entry_id": row["entry_id"], "missing": missing})
            if row.get("reviewer") and row["reviewer"].strip():
                reviewers.add(row["reviewer"].strip())
        if len(images) != manifest["sample_count"] or any(len(v) != 4 for v in images.values()):
            raise ValueError("Require four blinded outputs per image")
        if len(reviewers) > 1 or identities & reviewers:
            raise ValueError("Each rating file needs one distinct independent reviewer")
        identities |= reviewers
        progress[rater] = {"expected": len(rows), "complete": len(rows) - len(pending),
                           "remaining": len(pending),
                           "images_complete": sum(map(all, images.values())),
                           "pending_entries": pending,
                           "missing_counts": dict(Counter(
                               field for entry in pending for field in entry["missing"]))}
    if any(sha256_file(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("Review bundle or ratings changed during inspection")
    return {"schema": "blind_review_progress_v1", "split": manifest["split"],
            "bundle": str(bundle), "images": manifest["sample_count"],
            "status": "ready_to_score" if all(p["remaining"] == 0 for p in progress.values())
            else "awaiting_ratings", "raters": progress,
            "sources": {key: {"path": str(path), "sha256": hashes[key]}
                        for key, path in sources.items()},
            "summary": None, "agreement": None,
            "limitation": "Completeness checks do not establish authorship or independence."}


def save_review_progress(report, output):
    from vqa_explanation.evaluation.rubric_calibration import check_new_output

    sources = report["sources"].values()
    if any(sha256_file(Path(s["path"])) != s["sha256"] for s in sources):
        raise ValueError("Review bundle or ratings changed after inspection")
    output = check_new_output(output, [report["bundle"], *(s["path"] for s in sources)])
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    lines = ["# Tiến độ chấm theo ảnh", "", f"{report['images']} ảnh {report['split']}.", "",
             "| File nhãn | Đầu ra đủ trường | Ảnh đủ bốn đầu ra | Còn lại |",
             "|---|---:|---:|---:|"]
    for rater, row in report["raters"].items():
        lines.append(f"| {rater} | {row['complete']}/{row['expected']} "
                     f"| {row['images_complete']}/{report['images']} | {row['remaining']} |")
    lines += ["", "Chỉ kiểm tiến độ và nguồn tệp; chưa tính điểm hoặc độ đồng thuận.",
              "Giữ nhãn độc lập, dùng Xuất nháp sau mỗi phiên. Không sửa output gốc.", ""]
    (output / "README.md").write_text("\n".join(lines), "utf-8")
    return output


def refresh_review_pages(bundle, output):
    """Create a new bundle with current controls and unchanged blinded content."""
    from vqa_explanation.evaluation.rubric_calibration import check_new_output

    report = inspect_review_progress(bundle)
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    notes = {rater: list(read_jsonl(bundle / rater / "blank_ratings.jsonl"))
             for rater in manifest["raters"]}
    if any(row["review_status"] != "pending" or any(row.get(k) is not None
           for k in LABEL_FIELDS - {"review_status"})
           for rows in notes.values() for row in rows):
        raise ValueError("Refresh requires blank source forms; keep ratings in separate files")
    output = check_new_output(output, [bundle])
    for source in report["sources"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("Review bundle changed after inspection")
    output.mkdir(parents=True)
    pages = {f"{rater}/index.html" for rater in manifest["raters"]}
    for relative in manifest["files"]:
        if relative not in pages:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / relative, target)
            if sha256_file(target) != manifest["files"][relative]:
                raise ValueError("Review bundle changed while copying")
    for rater, rows in notes.items():
        (output / rater / "index.html").write_text(
            render_blind_editor(rows, split=manifest["split"]), "utf-8")
    manifest["origin_manifest_sha256"] = report["sources"]["bundle_manifest"]["sha256"]
    manifest["files"] = {p.relative_to(output).as_posix(): sha256_file(p)
                         for p in output.rglob("*") if p.is_file()}
    write_json(output / "private/manifest.json", manifest)
    inspect_review_progress(output)
    if any(sha256_file(Path(s["path"])) != s["sha256"] for s in report["sources"].values()):
        raise ValueError("Review bundle changed during refresh")
    return output
