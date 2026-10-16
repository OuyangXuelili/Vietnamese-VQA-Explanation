"""Export all four study arms in one inference environment."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from vqa_explanation.inference.export import read_jsonl, validate_selected_records
from vqa_explanation.inference.export_relocation import (
    prepare_relocation,
    prepare_snapshot_recovery,
    snapshot_export,
    verify_relocation_group,
    write_relocation,
)
from vqa_explanation.training.preference_data import load_config, validate_study_arm
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.process import run_command
from vqa_explanation.utils.run_provenance import sha256_file

VALIDATION_SHA256 = "7f0b2b7d374d386030ddea50bc57d3e8e6c76e257e9fc4d1203bfb331b5f5b6e"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ARMS = {"m0": "M0_PRETRAINED", "m1": "M1_SFT",
        "m2": "M2_SFT_CHOSEN", "m3": "M3_DPO"}


def verify_adapter(path, expected):
    for name, digest in expected.items():
        if name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError("Unsupported adapter evidence file")
        if sha256_file(path / name) != digest:
            raise ValueError(f"Adapter changed: {path / name}")
    if set(expected) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("Incomplete adapter evidence")


def main_study_settings(study_config):
    """Read the declared budget; never infer a new protocol from finished runs."""
    path = Path(study_config)
    study = yaml.safe_load(path.read_text("utf-8"))
    policy = study.get("preferences", {})
    count = policy.get("locked_pairs")
    if (study.get("protocol_id") != "vqa_nle_visual_preferences_v2"
            or policy.get("data_locked") is not True
            or policy.get("require_visual_quality") is not True
            or type(count) is not int or count < policy.get("minimum_pairs", math.inf)
            or policy.get("required_review_version") != 2):
        raise ValueError("Main validation requires the locked v2 study")
    result = {}
    for arm in ("m2", "m3"):
        config_path = project_root() / study["training"][f"{arm}_config"]
        cfg = load_config(config_path)
        protocol = validate_study_arm(config_path, path)
        result[arm] = {"config": cfg, "protocol": protocol, "pairs": count,
                       "epochs": cfg["num_train_epochs"],
                       "updates": math.ceil(count / protocol["effective_batch"]) * cfg[
                           "num_train_epochs"],
                       "quality_sha256": policy["quality_review_sha256"]}
    return result


def verify_main_stage(stage, arm, report, plan, settings, annotations_hash):
    quality = plan.get("visual_quality", {})
    if (plan.get("config") != settings["config"]
            or plan.get("comparison_protocol") != settings["protocol"]
            or plan.get("expected_updates") != settings["updates"]
            or plan.get("annotations_sha256") != annotations_hash
            or quality.get("sha256") != settings["quality_sha256"]
            or quality.get("ready") is not True
            or quality.get("confirmed") != settings["pairs"]
            or quality.get("unique_images") != settings["pairs"]
            or quality.get("required_review_version") != 2):
        raise ValueError(f"{arm}: training does not match the locked study")
    state = json.loads((stage / "training/trainer_state.json").read_text("utf-8"))
    if (state.get("global_step") != settings["updates"]
            or state.get("max_steps") != settings["updates"]
            or state.get("epoch") != settings["epochs"]
            or not isinstance(report.get("training_loss"), (int, float))
            or not math.isfinite(report["training_loss"])):
        raise ValueError(f"{arm}: final trainer state is incomplete")


def prepare_validation(environment_path, run_group, m1_adapter, output_dir, *, study_config=None):
    environment = load_environment(Path(environment_path))
    annotations = environment["annotations"].parent / "validation.jsonl"
    if sha256_file(annotations) != VALIDATION_SHA256:
        raise ValueError("Frozen validation annotations changed")
    records = read_jsonl(annotations)
    ids = {row["question_id"] for row in records}
    if len(records) != 1407 or len(ids) != 1407 or any(
        row.get("split") != "validation" for row in records
    ):
        raise ValueError("Require all 1407 unique validation questions")
    validate_selected_records(records, environment["image_root"])
    run_group, output_dir = Path(run_group).resolve(), Path(output_dir).resolve()
    protected = (run_group, environment["image_root"], project_root(), Path(m1_adapter).resolve())
    if any(path.is_relative_to(output_dir) or output_dir.is_relative_to(path)
           for path in protected):
        raise ValueError("Evaluation output cannot contain source, data or training runs")
    settings = main_study_settings(study_config) if study_config is not None else None
    canonical_hash = sha256_file(environment["annotations"]) if settings else None
    adapters, evidence, plans = {"m0": None}, {}, {}
    for arm in ("m2", "m3"):
        stage = run_group / f"{arm}_train"
        report = json.loads((stage / "training_result.json").read_text("utf-8"))
        plan = json.loads((stage / "data_plan.json").read_text("utf-8"))
        content = {key: value for key, value in plan.items() if key != "sha256"}
        plan_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if plan_hash != plan["sha256"]:
            raise ValueError(f"{arm}: training plan changed")
        budget = settings[arm] if settings else {"updates": 24, "epochs": 1, "pairs": 186}
        if (report.get("status") != "passed" or report.get("global_step") != budget["updates"]
                or report.get("epoch") != budget["epochs"]
                or report.get("policy_updated") is not True
                or report.get("saved_policy_verified") is not True
                or (arm == "m3" and report.get("reference_unchanged") is not True)):
            raise ValueError(f"{arm}: training is not complete")
        if (report.get("plan_sha256") != plan.get("sha256")
                or plan.get("pairs") != budget["pairs"]):
            raise ValueError(f"{arm}: training plan does not match report")
        if (plan["model"]["id"] != MODEL_ID
                or plan["model"]["revision"] != MODEL_REVISION):
            raise ValueError(f"{arm}: unexpected backbone")
        if settings:
            verify_main_stage(stage, arm, report, plan, budget, canonical_hash)
        adapter = stage / "training" / f"checkpoint-{budget['updates']}"
        verify_adapter(adapter, report["output_adapter"]["files"])
        adapters[arm], plans[arm] = adapter, plan
        evidence[arm] = {"report_sha256": sha256_file(stage / "training_result.json"),
                         "plan_sha256": plan["sha256"], "checkpoint": str(adapter),
                         "adapter_files": report["output_adapter"]["files"]}
    if (plans["m2"]["m1_adapter"] != plans["m3"]["m1_adapter"]
            or plans["m2"]["preferences_manifest_sha256"]
            != plans["m3"]["preferences_manifest_sha256"]):
        raise ValueError("The study arms have different M1 or preference inputs")
    if settings and any(plans["m2"].get(key) != plans["m3"].get(key) for key in (
        "annotations_sha256", "preferences_files", "m1_selection_sha256",
        "source_scope", "source_files", "visual_quality",
    )):
        raise ValueError("The main arms have different data, quality or training source")
    adapters["m1"] = Path(m1_adapter).resolve()
    verify_adapter(adapters["m1"], plans["m2"]["m1_adapter"]["files"])
    return environment, annotations, adapters, evidence


def comparison_cache(destination, left, right, annotations, *, record=False):
    """Reuse only a completed comparison bound to the same predictions and code."""
    receipt = destination.with_suffix(".receipt.json")
    identity = {
        "schema": "validation_comparison_receipt_v1",
        "inputs": {"left": sha256_file(left), "right": sha256_file(right),
                   "annotations": sha256_file(annotations)},
        "source": {name: sha256_file(project_root() / name) for name in (
            "scripts/compare_runs.py", "src/vqa_explanation/evaluation/comparison.py",
            "src/vqa_explanation/evaluation/answer_metrics.py",
            "src/vqa_explanation/evaluation/explanation_metrics.py",
        )},
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
    }
    if not destination.exists():
        if receipt.exists():
            raise ValueError("Comparison receipt exists without its report")
        return False
    report = json.loads(destination.read_text("utf-8"))
    if report.get("comparable") is not True:
        raise ValueError("Stored comparison is not comparable")
    identity["report_sha256"] = sha256_file(destination)
    if record:
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2)
    elif not receipt.is_file() or json.loads(receipt.read_text("utf-8")) != identity:
        raise ValueError("Stored comparison inputs, source or report changed")
    return True


def recover_validation(environment_path, run_group, m1_adapter, source_dir, output_dir, *,
                       study_config, inspect_only=False):
    """Seed a separate validation attempt from verified files and recovery copies."""
    environment, annotations, adapters, _ = prepare_validation(
        environment_path, run_group, m1_adapter, output_dir, study_config=study_config
    )
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if not source_dir.is_dir() or output_dir.exists():
        raise ValueError("Recovery requires an existing source and a new destination")
    plans = []
    for arm in ("m2", "m3", "m0", "m1"):
        args = SimpleNamespace(
            annotations=annotations, image_root=environment["image_root"],
            output=output_dir / f"{arm}.jsonl", adapter=adapters[arm], split="validation",
            limit=0, seed=42, model_id=MODEL_ID, revision=MODEL_REVISION,
            experiment_id=ARMS[arm], min_pixels=200704, max_pixels=200704,
            max_new_tokens=128, load_in_4bit=True,
        )
        plan = prepare_snapshot_recovery(source_dir / f"{arm}.jsonl", args)
        if plan is not None:
            plans.append(plan)
    if not plans:
        raise ValueError("No verified validation exports to recover")
    result = {"schema": "validation_snapshot_recovery_v1", "source_dir": str(source_dir),
              "output_dir": str(output_dir),
              "recovery_source_sha256": sha256_file(Path(__file__)),
              "snapshot_source_sha256": sha256_file(
                  Path(__file__).with_name("export_relocation.py")),
              "receipts": [plan["receipt"] for plan in plans]}
    if not inspect_only:
        result["receipts"] = [write_relocation(plan) for plan in plans]
        with (output_dir / "snapshot_recovery.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return {"status": "recovery_verified" if inspect_only else "recovered", **result}


def export_command(root, environment, annotations, adapter, arm, output):
    command = [sys.executable, "-u", str(root / "scripts/run_m0.py"),
               "--annotations", str(annotations), "--image-root", str(environment["image_root"]),
               "--output", str(output), "--split", "validation", "--limit", "0", "--seed", "42",
               "--model-id", MODEL_ID, "--revision", MODEL_REVISION,
               "--min-pixels", "200704", "--max-pixels", "200704",
               "--max-new-tokens", "128", "--load-in-4bit", "--experiment-id", ARMS[arm]]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if output.exists() or output.with_suffix(".jsonl.manifest.json").exists():
        command.append("--resume")
    return command


def run_validation(environment_path, run_group, m1_adapter, output_dir, *, inspect_only=False,
                   runner=run_command, relocate_from=None, study_config=None):
    environment, annotations, adapters, evidence = prepare_validation(
        environment_path, run_group, m1_adapter, output_dir, study_config=study_config
    )
    root, output_dir = project_root(), Path(output_dir).resolve()
    relocations = []
    if relocate_from is not None and not Path(relocate_from).is_dir():
        raise ValueError("Original export directory is missing")
    if relocate_from is not None and output_dir.exists():
        verify_relocation_group(relocate_from, output_dir)
    elif relocate_from is not None:
        for arm in ("m2", "m3", "m0", "m1"):
            args = SimpleNamespace(
                annotations=annotations, image_root=environment["image_root"],
                output=output_dir / f"{arm}.jsonl", adapter=adapters[arm],
                split="validation", limit=0, seed=42, model_id=MODEL_ID,
                revision=MODEL_REVISION, experiment_id=ARMS[arm],
                min_pixels=200704, max_pixels=200704, max_new_tokens=128, load_in_4bit=True,
            )
            plan = prepare_relocation(Path(relocate_from) / f"{arm}.jsonl", args)
            if plan is not None:
                relocations.append(plan)
        if not relocations:
            raise ValueError("Original directory has no verifiable exports")
        if not inspect_only:
            receipts = []
            for plan in relocations:
                receipt = write_relocation(plan)
                receipts.append(receipt)
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
            with (output_dir / "relocation_group.json").open("x", encoding="utf-8") as handle:
                json.dump({"schema": "validation_export_relocation_v1",
                           "source_dir": str(Path(relocate_from).resolve()),
                           "receipts": receipts}, handle, ensure_ascii=False, indent=2)
    commands = [export_command(root, environment, annotations, adapters[arm], arm,
                               output_dir / f"{arm}.jsonl") for arm in ("m2", "m3", "m0", "m1")]
    if inspect_only:
        return {"status": "inputs_verified", "questions": 1407, "commands": commands,
                "relocations": [plan["receipt"] for plan in relocations]}
    records = read_jsonl(annotations)
    for arm, command in zip(("m2", "m3", "m0", "m1"), commands, strict=True):
        print(f"Export {arm}: all 1407 validation questions", flush=True)

        def save_progress(line, output=output_dir / f"{arm}.jsonl"):
            match = re.fullmatch(r"\[(\d+)/1407\] \S+ format=(True|False)\n", line)
            if match and (int(match[1]) == 25 or int(match[1]) % 100 == 0):
                snapshot = snapshot_export(output, records)
                if snapshot:
                    print(f"Recovery snapshot: {snapshot['predictions']} predictions", flush=True)

        runner(command, cwd=root, on_output=save_progress)
    for left, right in (("m2", "m3"), ("m1", "m2"), ("m1", "m3"), ("m0", "m1")):
        destination = output_dir / f"{left}_{right}_comparison.json"
        predictions = (output_dir / f"{left}.jsonl", output_dir / f"{right}.jsonl")
        if study_config is not None and comparison_cache(destination, *predictions, annotations):
            continue
        if destination.exists():
            raise ValueError(f"Comparison already exists; preserve it: {destination}")
        runner([sys.executable, "-u", str(root / "scripts/compare_runs.py"),
                str(output_dir / f"{left}.jsonl"), str(output_dir / f"{right}.jsonl"),
                "--output", str(destination), "--left-experiment", ARMS[left],
                "--right-experiment", ARMS[right], "--expected-validation-annotations",
                str(annotations), "--expected-validation-sha256", VALIDATION_SHA256,
                "--bootstrap-resamples", "2000"], cwd=root)
        if study_config is not None:
            comparison_cache(destination, *predictions, annotations, record=True)
    return {"status": "completed", "questions": 1407, "training_evidence": evidence,
            "output_dir": str(output_dir)}
