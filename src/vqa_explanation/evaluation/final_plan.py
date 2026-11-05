"""Audit the eight precommitted test identities without loading a model."""

import json
import math
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.blind_review import select_sample
from vqa_explanation.evaluation.comparison import PROTOCOL_FIELDS
from vqa_explanation.evaluation.rubric_calibration import check_new_output, verified_validation
from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    DEFAULT_BERTSCORE_REVISION,
)
from vqa_explanation.inference.study_validation import ARMS
from vqa_explanation.utils.run_provenance import sha256_file

SEEDS = (42, 43, 44)


def build_final_plan(evidence_by_seed, studies_by_seed, annotations, sample_path, calibration):
    if set(evidence_by_seed) != set(SEEDS) or set(studies_by_seed) != set(SEEDS):
        raise ValueError("Require exactly the precommitted training seeds 42, 43 and 44")
    annotations, sample_path, calibration = map(Path, (annotations, sample_path, calibration))
    canonical = list(read_jsonl(annotations))
    if len({r["question_id"] for r in canonical}) != len(canonical):
        raise ValueError("Duplicate canonical question ID")
    splits = {split: [r for r in canonical if r["split"] == split]
              for split in ("train", "validation", "test")}
    if {split: len(rows) for split, rows in splits.items()} != {
        "train": 29459, "validation": 1407, "test": 1968
    } or sum(map(len, splits.values())) != len(canonical):
        raise ValueError("Canonical split sizes differ from the frozen study")
    images = {split: {r["image_id"] for r in rows} for split, rows in splits.items()}
    if any(images[a] & images[b] for a, b in (("train", "validation"), ("train", "test"),
                                             ("validation", "test"))):
        raise ValueError("Image leakage between canonical splits")
    test_annotations = annotations.parent / "test.jsonl"
    if sorted(read_jsonl(test_annotations), key=lambda r: r["question_id"]) != sorted(
        splits["test"], key=lambda r: r["question_id"]
    ):
        raise ValueError("Test annotations differ from canonical annotations")
    sample = json.loads(sample_path.read_text("utf-8"))
    selected = select_sample(canonical, count=200, seed=42)
    if (sample.get("schema") != "blind_review_sample_v1" or sample.get("split") != "test"
            or sample.get("count") != 200 or sample.get("seed") != 42
            or sample.get("annotations_sha256") != sha256_file(annotations)
            or sample.get("question_ids") != [r["question_id"] for r in selected]
            or sample.get("image_ids") != [r["image_id"] for r in selected]):
        raise ValueError("The preselected 200-image test sample changed")
    calibration_manifest = calibration / "manifest.json"
    practice = json.loads(calibration_manifest.read_text("utf-8"))
    if (practice.get("schema") != "rubric_calibration_v1"
            or practice.get("status") != "awaiting_ratings"
            or practice.get("split") != "validation" or practice.get("images") != 30
            or practice.get("outputs_per_rater") != 120 or practice.get("training_seed") != 42):
        raise ValueError("Require the separate validation rubric calibration")
    for relative, digest in practice["files"].items():
        path = (calibration / relative).resolve()
        if not path.is_relative_to(calibration.resolve()) or sha256_file(path) != digest:
            raise ValueError("Calibration files changed")
    runs, protocol, inference_source, baselines, sources = {}, None, None, {}, {}
    validation = annotations.parent / "validation.jsonl"
    for seed in SEEDS:
        folder, study = Path(evidence_by_seed[seed]), Path(studies_by_seed[seed])
        paths, predictions, evidence = verified_validation(folder, validation, annotations, study)
        if evidence["training_seed"] != seed or not math.isclose(
            evidence["training_learning_rate"], 1e-5, rel_tol=1e-12
        ):
            raise ValueError("Main test identities must not be replaced by LR sensitivity runs")
        for arm in ARMS:
            manifest = evidence["manifests"][arm]
            current = {field: predictions[arm][0][field] for field in PROTOCOL_FIELDS}
            current_source = manifest["inputs"]["source_files"]
            if protocol is None:
                protocol, inference_source = current, current_source
            if current != protocol or current_source != inference_source:
                raise ValueError("Inference protocol/source differs across the main runs")
            if seed == 42 and practice["sources"][arm]["sha256"] != sha256_file(paths[arm]):
                raise ValueError("Calibration does not use the main seed 42 outputs")
            identity = {"experiment_id": ARMS[arm],
                        "training_seed": None if arm == "m0" else "M1_frozen"
                        if arm == "m1" else seed,
                        "adapter_at_validation": manifest["inputs"]["adapter"],
                        "adapter_files": manifest["inputs"]["adapter_files"]}
            if arm in {"m0", "m1"}:
                if arm in baselines and identity != baselines[arm]:
                    raise ValueError("M0/M1 identity differs between training seeds")
                baselines[arm] = identity
                runs[arm] = identity
            else:
                runs[f"{arm}_seed{seed}"] = identity
        files = {"study_config": study, "m1_selection": folder / "m1_selection.json", **paths}
        files.update({f"{arm}_manifest": path.with_suffix(".jsonl.manifest.json")
                      for arm, path in paths.items()})
        files.update({p.relative_to(folder).as_posix(): p
                      for p in (folder / "training").rglob("*.json")})
        sources[str(seed)] = {key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                              for key, path in files.items()}
    return {
        "schema": "final_test_preparation_v1", "status": "pending_rubric_and_runtime",
        "split": "test", "questions_per_run": 1968, "test_images": len(images["test"]),
        "runs": runs, "expected_predictions": 1968 * len(runs),
        "inference_protocol": protocol, "inference_source": inference_source,
        "inference_order_seed": 42, "training_seeds": list(SEEDS),
        "human_review": {"images": 200, "representative_training_seed": 42,
                         "runs": ["m0", "m1", "m2_seed42", "m3_seed42"],
                         "sample_sha256": sha256_file(sample_path),
                         "calibration_manifest_sha256": sha256_file(calibration_manifest)},
        "metrics": {"answer": ["normalized_exact_match", "whitespace_token_f1"],
                    "explanation": ["rouge_l_f1", "bertscore_f1"],
                    "bertscore_model": DEFAULT_BERTSCORE_MODEL,
                    "bertscore_revision": DEFAULT_BERTSCORE_REVISION,
                    "bertscore_layers": DEFAULT_BERTSCORE_LAYERS,
                    "bootstrap_resamples": 2000, "bootstrap_seed": 42,
                    "main_contrast": "M3_minus_M2", "keep_all_outputs": True},
        "annotations_sha256": sha256_file(annotations),
        "test_annotations_sha256": sha256_file(test_annotations), "sources": sources,
        "adapter_weights_verified_in_this_step": False,
        "test_predictions_generated": False,
        "blockers": ["Complete validation rubric calibration and freeze its instructions.",
                     "Verify actual adapter weights and pinned inference runtime on GPU."],
    }


def save_final_plan(plan, output):
    output = check_new_output(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    (output / "README.md").write_text(
        "# Chuẩn bị đánh giá test\n\n"
        "Đã đối chiếu hồ sơ train và validation của ba seed chính. "
        "Có tám identity: M0, M1 và M2/M3 của seed 42/43/44. "
        "Mỗi run cần 1.968 câu, tổng cộng 15.744 dự đoán.\n\n"
        "Đây chưa phải lệnh chạy hoặc kết quả test. Đường dẫn adapter trong hồ sơ là "
        "đường dẫn của lượt validation, không tự coi là đường dẫn runtime hiện tại. "
        "Cần hoàn tất chấm thử 30 ảnh, khóa rubric và kiểm trọng số/môi trường GPU "
        "trước khi triển khai export. Không chọn lại seed hoặc LR bằng test.\n", "utf-8")
    write_json(output / "manifest.json", {"files": {
        p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    return output
