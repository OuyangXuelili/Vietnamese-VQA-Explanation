"""Freeze instructions before test inference without inventing human ratings."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.evaluation.rubric_calibration import check_new_output, load_rubric
from vqa_explanation.utils.run_provenance import sha256_file

GUIDELINE_BLOCKER = "Freeze evaluation instructions; human calibration remains pending."
FROZEN_FILES = {
    "parent_plan.json", "plan.json", "rubric.yaml", "guidelines.md",
    "calibration_manifest.json", "reference_manifest.json", "reference_summary.json",
    "reference_assessments.jsonl",
}


def verify_frozen_guidelines(directory, plan):
    directory = Path(directory)
    lock = json.loads((directory / "lock.json").read_text("utf-8"))
    amendment = plan.get("sequence_amendment", {})
    if (lock.get("schema") != "frozen_evaluation_guidelines_v1"
            or lock.get("status") != "frozen" or lock.get("human_calibration_complete") is not False
            or lock.get("human_evaluation_complete") is not False or lock.get("human_raters") != 0
            or set(lock.get("files", {})) != FROZEN_FILES
            or amendment.get("schema") != "pretest_sequence_amendment_v1"
            or amendment.get("change") != "freeze_instructions_before_human_calibration"
            or amendment.get("human_review_required") is not True
            or amendment.get("human_calibration_complete") is not False
            or amendment.get("test_predictions_before_declaration") != 0):
        raise ValueError("Require a pretest instruction freeze with human ratings pending")
    for name, digest in lock["files"].items():
        path = directory / name
        if path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"Frozen evaluation instructions changed: {name}")
    stored_plan = json.loads((directory / "plan.json").read_text("utf-8"))
    parent = json.loads((directory / "parent_plan.json").read_text("utf-8"))
    expected = copy.deepcopy(parent)
    expected["sequence_amendment"] = amendment
    expected["blockers"] = [GUIDELINE_BLOCKER, *parent.get("blockers", [])[1:]]
    if (stored_plan != plan or plan != expected
            or amendment.get("parent_plan_sha256") != sha256_file(directory / "parent_plan.json")
            or amendment.get("calibration_manifest_sha256")
            != plan["human_review"]["calibration_manifest_sha256"]
            or amendment["calibration_manifest_sha256"]
            != sha256_file(directory / "calibration_manifest.json")
            or amendment.get("reference_manifest_sha256")
            != sha256_file(directory / "reference_manifest.json")
            or amendment.get("guidelines_sha256") != sha256_file(directory / "guidelines.md")
            or amendment.get("rubric_sha256") != sha256_file(directory / "rubric.yaml")):
        raise ValueError("The amendment may change evaluation sequencing only")
    declared = datetime.fromisoformat(amendment["declared_at_utc"])
    if declared.utcoffset() is None:
        raise ValueError("The pretest declaration requires a timezone")
    load_rubric(directory / "rubric.yaml")
    summary = json.loads((directory / "reference_summary.json").read_text("utf-8"))
    reference_files = json.loads(
        (directory / "reference_manifest.json").read_text("utf-8"))["files"]
    if (reference_files.get("summary.json") != sha256_file(directory / "reference_summary.json")
            or reference_files.get("reference_assessments.jsonl")
            != sha256_file(directory / "reference_assessments.jsonl")):
        raise ValueError("Frozen reference differs from its original manifest")
    rows = list(read_jsonl(directory / "reference_assessments.jsonl"))
    if (summary.get("assessment_method") != "assistant_visual_review"
            or summary.get("human_ratings") is not False or summary.get("split") != "validation"
            or summary.get("images") != 30 or summary.get("outputs") != 120
            or len(rows) != 120 or len({r["entry_id"] for r in rows}) != 120
            or len({r["image_index"] for r in rows}) != 30):
        raise ValueError("Require the complete validation reference, not human calibration results")
    return sha256_file(directory / "lock.json")


def freeze_guidelines(plan_path, plan_sha256, calibration, reference, guidelines,
                      annotations, test_runs_root, output):
    from vqa_explanation.evaluation.test_report import inspect_test_outputs
    from vqa_explanation.inference.study_test import IDENTITIES, read_test_plan

    plan_path, calibration, reference, guidelines = map(
        Path, (plan_path, calibration, reference, guidelines))
    test_runs_root = Path(test_runs_root)
    output = check_new_output(output, (plan_path, calibration, reference, guidelines,
                                       annotations, test_runs_root))
    plan = read_test_plan(plan_path, plan_sha256)
    if "sequence_amendment" in plan:
        raise ValueError("Do not amend an already amended test plan")
    _, _, _, progress, _ = inspect_test_outputs(
        plan_path, plan_sha256, annotations,
        {identity: test_runs_root / identity for identity in IDENTITIES})
    if (any(row["status"] != "missing" for row in progress.values())
            or (test_runs_root.exists() and any(test_runs_root.rglob("*.jsonl")))):
        raise ValueError("Freeze instructions before any test export is prepared or generated")
    calibration_manifest = calibration / "manifest.json"
    if sha256_file(calibration_manifest) != plan["human_review"]["calibration_manifest_sha256"]:
        raise ValueError("Calibration differs from the original test plan")
    for folder in (calibration, reference):
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        for relative, digest in manifest.get("files", manifest).items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder.resolve()) or sha256_file(path) != digest:
                raise ValueError("Validation reference or calibration changed")
    rubric = calibration / "rubric.yaml"
    load_rubric(rubric)
    if len(guidelines.read_text("utf-8").strip()) < 200:
        raise ValueError("Record decisions for the ambiguous validation cases")
    expected = {r["entry_id"]: r for r in read_jsonl(
        calibration / "bundle/rater1/blank_ratings.jsonl")}
    rows = list(read_jsonl(reference / "reference_assessments.jsonl"))
    if len(rows) != 120 or {r["entry_id"] for r in rows} != set(expected):
        raise ValueError("Reference does not cover the validation practice entries")
    for row in rows:
        original = expected[row["entry_id"]]
        if any(row[k] != original[k] for k in (
                "question", "prediction_answer", "prediction_explanation")):
            raise ValueError("Reference changed a question or model output")
        before = (calibration / "bundle/rater1" / original["image_path"]).resolve()
        after = (reference / row["image_path"]).resolve()
        if (not before.is_relative_to(calibration.resolve())
                or not after.is_relative_to(reference.resolve())
                or sha256_file(before) != sha256_file(after)):
            raise ValueError("Reference changed a validation image")
    sources = {
        "parent_plan.json": plan_path, "rubric.yaml": rubric, "guidelines.md": guidelines,
        "calibration_manifest.json": calibration_manifest,
        "reference_manifest.json": reference / "manifest.json",
        "reference_summary.json": reference / "summary.json",
        "reference_assessments.jsonl": reference / "reference_assessments.jsonl",
    }
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    amended = copy.deepcopy(plan)
    amended["sequence_amendment"] = {
        "schema": "pretest_sequence_amendment_v1",
        "change": "freeze_instructions_before_human_calibration",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_plan_sha256": plan_sha256,
        "calibration_manifest_sha256": hashes["calibration_manifest.json"],
        "reference_manifest_sha256": hashes["reference_manifest.json"],
        "rubric_sha256": hashes["rubric.yaml"], "guidelines_sha256": hashes["guidelines.md"],
        "human_calibration_complete": False, "human_review_required": True,
        "test_predictions_before_declaration": 0,
    }
    amended["blockers"] = [GUIDELINE_BLOCKER, *plan.get("blockers", [])[1:]]
    output.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        content = path.read_bytes()
        if sha256_file(path) != hashes[name]:
            raise ValueError("Guideline freeze inputs changed")
        (output / name).write_bytes(content)
    write_json(output / "plan.json", amended)
    lock = {"schema": "frozen_evaluation_guidelines_v1", "status": "frozen",
            "human_calibration_complete": False, "human_evaluation_complete": False,
            "human_raters": 0,
            "files": {name: sha256_file(output / name) for name in sorted(FROZEN_FILES)}}
    write_json(output / "lock.json", lock)
    verify_frozen_guidelines(output, amended)
    return {"status": "instructions_frozen_human_calibration_pending",
            "plan_sha256": sha256_file(output / "plan.json"),
            "lock_sha256": sha256_file(output / "lock.json"), "human_ratings": 0}
