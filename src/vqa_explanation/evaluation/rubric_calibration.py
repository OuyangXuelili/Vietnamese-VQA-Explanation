"""Prepare validation-only practice pages and freeze a completed rating rubric."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.blind_review import (
    freeze_sample,
    prepare_blind_review,
    summarize_blind_review,
)
from vqa_explanation.evaluation.study_report import verify_evidence
from vqa_explanation.inference.study_validation import ARMS, VALIDATION_SHA256
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def load_rubric(path):
    rubric = yaml.safe_load(Path(path).read_text("utf-8"))
    if (not isinstance(rubric, dict)
            or rubric.get("protocol_id") != "vqa_nle_image_support_v1"
            or rubric.get("revision") != 1 or rubric.get("calibration_images") != 30
            or rubric.get("test_images") != 200
            or rubric.get("representative_training_seed") != 42
            or rubric.get("primary_contrast") != "M3_minus_M2"
            or rubric.get("primary_metric") != "explanation_support"):
        raise ValueError("Rubric differs from the declared evaluation design")
    for field, labels in (("answer_assessment", {"correct", "incorrect", "uncertain"}),
                          ("explanation_support", {0, 1, 2}),
                          ("unsupported_detail", {"yes", "no", "uncertain"})):
        values = rubric.get(field)
        if (not isinstance(values, dict) or set(values) != labels
                or any(not isinstance(text, str) or not text.strip() for text in values.values())):
            raise ValueError(f"Incomplete rubric: {field}")
    instructions = rubric.get("instructions")
    if (not isinstance(instructions, list) or not instructions
            or any(not isinstance(text, str) or not text.strip() for text in instructions)):
        raise ValueError("Rubric instructions are required")
    return rubric


def check_new_output(output, protected=()):
    output = Path(output).resolve()
    root = project_root().resolve()
    paths = [root / name for name in ("src", "scripts", "tests", "configs", "notebooks",
                                     "app", "data", "runs", ".git")]
    paths += [Path(path).resolve() for path in protected]
    if (output == root or root.is_relative_to(output)
            or any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths)):
        raise ValueError("Output overlaps protected inputs or source")
    if output.exists():
        raise FileExistsError(output)
    return output


def verified_validation(folder, annotations_path, canonical_path, study_config):
    annotations_path, canonical_path = Path(annotations_path), Path(canonical_path)
    if sha256_file(annotations_path) != VALIDATION_SHA256:
        raise ValueError("Frozen validation annotations changed")
    annotations = list(read_jsonl(annotations_path))
    if len(annotations) != 1407 or len({r["question_id"] for r in annotations}) != 1407:
        raise ValueError("Require all 1,407 unique validation questions")
    canonical = [r for r in read_jsonl(canonical_path) if r["split"] == "validation"]
    if sorted(canonical, key=lambda r: r["question_id"]) != sorted(
        annotations, key=lambda r: r["question_id"]
    ):
        raise ValueError("Validation differs from canonical annotations")
    folder = Path(folder)
    paths = {arm: folder / "validation" / f"{arm}.jsonl" for arm in ARMS}
    predictions = {arm: list(read_jsonl(path)) for arm, path in paths.items()}
    evidence = verify_evidence(folder, annotations, predictions, study_config, canonical_path)
    return paths, predictions, evidence


def prepare_calibration(folder, annotations, canonical, image_root, study_config, rubric, output):
    folder, rubric = Path(folder), Path(rubric)
    output = check_new_output(output, (folder, annotations, canonical, image_root, rubric))
    load_rubric(rubric)
    paths, _, evidence = verified_validation(folder, annotations, canonical, study_config)
    if evidence["training_seed"] != 42 or not math.isclose(
        evidence["training_learning_rate"], 1e-5, rel_tol=1e-12
    ):
        raise ValueError("Calibration uses the precommitted main seed 42, not LR sensitivity")
    sources = {"annotations": Path(annotations), "canonical": Path(canonical),
               "rubric": rubric, "study_config": Path(study_config), **paths}
    for path in (folder / "training").rglob("*.json"):
        sources[path.relative_to(folder).as_posix()] = path
    sources["m1_selection"] = folder / "m1_selection.json"
    sources.update({f"{arm}_manifest": path.with_suffix(".jsonl.manifest.json")
                    for arm, path in paths.items()})
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    sample = output / "sample.json"
    freeze_sample(canonical, sample, count=30, seed=42, split="validation")
    prepare_blind_review(canonical, sample, {ARMS[arm]: path for arm, path in paths.items()},
                         image_root, output / "bundle", expected_split="validation")
    (output / "rubric.yaml").write_bytes(rubric.read_bytes())
    (output / "README.md").write_text(
        "# Thống nhất cách chấm\n\n"
        "30 ảnh validation, mỗi ảnh một câu và bốn đầu ra đã lưu của lượt chính seed 42. "
        "Không sinh lại đáp án, không mở test, không gán nhãn sẵn.\n\n"
        "1. Đọc `rubric.yaml`, rồi mở `bundle/rater1/index.html` hoặc "
        "`bundle/rater2/index.html`. Mỗi người có 120 đầu ra.\n"
        "2. Chấm độc lập bằng mã ẩn danh, xuất JSONL để giữ nhãn trước thảo luận.\n"
        "3. Thảo luận điểm chưa thống nhất, sửa hướng dẫn nếu cần, rồi khóa rubric. "
        "Nhãn gốc không sửa hoặc ghi đè.\n\n"
        "Chỉ gửi thư mục người chấm và ảnh cho người đó; không gửi `bundle/private`. "
        "Đây là bước luyện tiêu chí, không phải kết quả chất lượng trên test.\n", "utf-8")
    if any(sha256_file(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("Calibration inputs changed during preparation")
    manifest = {"schema": "rubric_calibration_v1", "status": "awaiting_ratings",
                "split": "validation", "images": 30, "outputs_per_rater": 120,
                "training_seed": 42, "training_learning_rate": 1e-5,
                "sources": {key: {"path": str(path.resolve()), "sha256": hashes[key]}
                            for key, path in sources.items()},
                "files": {path.relative_to(output).as_posix(): sha256_file(path)
                          for path in output.rglob("*") if path.is_file()}}
    write_json(output / "manifest.json", manifest)
    return manifest


def freeze_rubric(calibration, rating_paths, rubric, output):
    calibration, rubric = Path(calibration), Path(rubric)
    output = check_new_output(output, (calibration, rubric, *rating_paths.values()))
    load_rubric(rubric)
    manifest = json.loads((calibration / "manifest.json").read_text("utf-8"))
    if (manifest.get("schema") != "rubric_calibration_v1"
            or manifest.get("split") != "validation" or manifest.get("images") != 30
            or manifest.get("outputs_per_rater") != 120):
        raise ValueError("Require the 30-image validation calibration")
    for name, digest in manifest["files"].items():
        path = (calibration / name).resolve()
        if not path.is_relative_to(calibration.resolve()) or sha256_file(path) != digest:
            raise ValueError("Calibration bundle changed")
    summary = summarize_blind_review(calibration / "bundle", rating_paths)
    if (summary["status"] != "calibration_complete"
            or any(stats[role]["n"] != 30 for stats in summary["raters"].values()
                   for role in ARMS.values())):
        raise ValueError("Require complete validation ratings before locking the rubric")
    output.mkdir(parents=True, exist_ok=False)
    (output / "rubric.yaml").write_bytes(rubric.read_bytes())
    write_json(output / "calibration_summary.json", summary)
    record = {"schema": "frozen_human_rubric_v1", "status": "frozen",
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "calibration_manifest_sha256": sha256_file(calibration / "manifest.json"),
              "rubric_sha256": sha256_file(output / "rubric.yaml"),
              "summary_sha256": sha256_file(output / "calibration_summary.json"),
              "raters": len(rating_paths),
              "ratings": {key: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                          for key, path in rating_paths.items()},
              "limitation": "Metadata cannot prove that ratings were made independently by people."}
    write_json(output / "lock.json", record)
    return record
