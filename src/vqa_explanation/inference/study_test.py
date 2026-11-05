"""Run one frozen test identity with the existing prediction exporter."""

import json
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock

from vqa_explanation.data.io import write_json
from vqa_explanation.evaluation.comparison import PROTOCOL_FIELDS
from vqa_explanation.evaluation.rubric_calibration import check_new_output, load_rubric
from vqa_explanation.inference.export import (
    export_inputs,
    export_predictions,
    read_jsonl,
    select_records,
    validate_saved_predictions,
    validate_selected_records,
)
from vqa_explanation.inference.export_relocation import (
    prepare_relocation,
    require_safe_snapshot_resume,
    snapshot_export,
    verified_snapshots,
    write_relocation,
)
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner
from vqa_explanation.inference.study_validation import (
    ARMS,
    MODEL_ID,
    MODEL_REVISION,
    verify_adapter,
)
from vqa_explanation.training.image_integrity import verify_record_images
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.run_provenance import sha256_file

IDENTITIES = {"m0": ("m0", None), "m1": ("m1", "M1_frozen"),
              **{f"{arm}_seed{seed}": (arm, seed)
                 for seed in (42, 43, 44) for arm in ("m2", "m3")}}


def read_test_plan(path, expected_sha256):
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Final test plan SHA-256 mismatch")
    plan = json.loads(path.read_text("utf-8"))
    if (plan.get("schema") != "final_test_preparation_v1" or plan.get("split") != "test"
            or plan.get("questions_per_run") != 1968 or plan.get("expected_predictions") != 15744
            or set(plan.get("runs", {})) != set(IDENTITIES)):
        raise ValueError("Require the frozen eight-identity test plan")
    protocol = plan["inference_protocol"]
    if (set(protocol) != set(PROTOCOL_FIELDS) or protocol["model_id"] != MODEL_ID
            or protocol["model_revision"] != MODEL_REVISION or protocol["seed"] != 42
            or plan.get("inference_order_seed") != 42):
        raise ValueError("Unexpected frozen inference protocol")
    for name, (arm, seed) in IDENTITIES.items():
        identity = plan["runs"][name]
        if identity["experiment_id"] != ARMS[arm] or identity["training_seed"] != seed:
            raise ValueError(f"Unexpected test identity: {name}")
        files = identity["adapter_files"]
        if (arm == "m0" and files is not None) or (
            arm != "m0" and set(files or {}) != {
                "adapter_config.json", "adapter_model.safetensors"}
        ):
            raise ValueError(f"Incomplete frozen adapter identity: {name}")
    root = project_root().resolve()
    for name, digest in plan["inference_source"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or sha256_file(source) != digest:
            raise ValueError(f"Inference source differs from validation: {name}")
    return plan


def verify_rubric(directory, plan):
    if "sequence_amendment" in plan:
        from vqa_explanation.evaluation.guidelines import verify_frozen_guidelines

        return verify_frozen_guidelines(directory, plan)
    directory = Path(directory)
    lock = json.loads((directory / "lock.json").read_text("utf-8"))
    if (lock.get("schema") != "frozen_human_rubric_v1" or lock.get("status") != "frozen"
            or lock.get("calibration_manifest_sha256")
            != plan["human_review"]["calibration_manifest_sha256"]
            or lock.get("rubric_sha256") != sha256_file(directory / "rubric.yaml")
            or lock.get("summary_sha256") != sha256_file(directory / "calibration_summary.json")):
        raise ValueError("Require a frozen rubric from the planned validation calibration")
    load_rubric(directory / "rubric.yaml")
    summary = json.loads((directory / "calibration_summary.json").read_text("utf-8"))
    if summary.get("status") != "calibration_complete" or summary.get("split") != "validation":
        raise ValueError("Require completed validation calibration")
    raters = summary.get("raters", {})
    if (not raters or set(raters) != set(lock.get("ratings", {}))
            or lock.get("raters") != len(raters)
            or any(not set(ARMS.values()).issubset(stats)
                   or set(stats) - {*ARMS.values(), "M3_minus_M2"}
                   or any(stats[arm].get("n") != 30 for arm in ARMS.values())
                   for stats in raters.values())):
        raise ValueError("Rubric calibration is incomplete")
    return sha256_file(directory / "lock.json")


def checked_protocol(metadata, protocol):
    actual = {**{key: protocol[key] for key in ("model_id", "model_revision", "seed")},
              **metadata}
    if {key: actual.get(key) for key in PROTOCOL_FIELDS} != protocol:
        raise ValueError("Loaded inference protocol differs from frozen validation")


def build_test_run_record(args, plan, integrity):
    return {"schema": "final_test_run_v1", **args.export_context,
            "adapter": str(args.adapter) if args.adapter else None,
            "adapter_files": plan["runs"][args.export_context["test_identity"]]["adapter_files"],
            "image_integrity": integrity,
            "orchestration_source": {name: sha256_file(project_root() / name) for name in (
                "src/vqa_explanation/inference/study_test.py", "scripts/run_study_test.py")}}


def verify_test_completion(output_dir, identity):
    """Check a closed export before any attempt to resume or score it."""
    output_dir = Path(output_dir)
    closed = output_dir / "completion.json"
    if not closed.exists():
        return None
    record = json.loads(closed.read_text("utf-8"))
    output = output_dir / f"{identity}.jsonl"
    files = {"predictions_sha256": output,
             "manifest_sha256": output.with_suffix(".jsonl.manifest.json"),
             "run_sha256": output_dir / "test_run.json"}
    if (record.get("identity") != identity or record.get("predictions") != 1968
            or any(not path.is_file() or sha256_file(path) != record.get(key)
                   for key, path in files.items())):
        raise ValueError("Completed test export changed; preserve it before recovery")
    return record


def prepare_test(environment_path, plan_path, plan_sha256, identity, output_dir, *,
                 adapter=None, rubric=None):
    plan = read_test_plan(plan_path, plan_sha256)
    if identity not in IDENTITIES:
        raise ValueError(f"Unknown test identity: {identity}")
    environment = load_environment(Path(environment_path))
    canonical_path = environment["annotations"]
    annotations = canonical_path.parent / "test.jsonl"
    if (sha256_file(canonical_path) != plan["annotations_sha256"]
            or sha256_file(annotations) != plan["test_annotations_sha256"]):
        raise ValueError("Frozen test annotations changed")
    records = read_jsonl(annotations)
    canonical = [row for row in read_jsonl(canonical_path) if row["split"] == "test"]
    if (len(records) != 1968 or any(row["split"] != "test" for row in records)
            or len({row["image_id"] for row in records}) != plan["test_images"]
            or sorted(records, key=lambda r: r["question_id"])
            != sorted(canonical, key=lambda r: r["question_id"])):
        raise ValueError("Test rows differ from the frozen canonical split")
    validate_selected_records(records, environment["image_root"])
    if "image_checksums" not in environment:
        raise ValueError("Test evaluation requires the fixed image checksum manifest")
    integrity = verify_record_images(records, environment["image_root"],
                                     environment["image_checksums"])
    selected = plan["runs"][identity]
    adapter = Path(adapter).resolve() if adapter is not None else None
    protected = [canonical_path.parent, environment["image_root"], environment["run_root"],
                 Path(plan_path).resolve().parent]
    protected += [p for p in (adapter, Path(rubric).resolve() if rubric else None) if p]
    output_dir = Path(output_dir).resolve()
    try:
        check_new_output(output_dir, protected)
    except FileExistsError:
        if not output_dir.is_dir() or not (output_dir / "test_run.json").is_file():
            raise ValueError("Existing test output has no matching run record") from None
    blockers = []
    if identity == "m0" and adapter is not None:
        raise ValueError("M0 must not load an adapter")
    if identity != "m0":
        if adapter is None:
            blockers.append("Provide the adapter for this test identity.")
        else:
            verify_adapter(adapter, selected["adapter_files"])
    rubric_hash = None
    if rubric is None or not (Path(rubric) / "lock.json").is_file():
        blockers.append("Complete validation calibration and freeze the rubric.")
    else:
        rubric_hash = verify_rubric(rubric, plan)
    protocol = plan["inference_protocol"]
    args = SimpleNamespace(
        annotations=annotations, image_root=environment["image_root"],
        output=output_dir / f"{identity}.jsonl", adapter=adapter,
        split="test", limit=0, experiment_id=selected["experiment_id"],
        **{key: protocol[key] for key in ("seed", "min_pixels", "max_pixels",
                                         "max_new_tokens", "load_in_4bit", "model_id")},
        revision=protocol["model_revision"],
        export_context={"test_identity": identity, "training_seed": selected["training_seed"],
                        "plan_sha256": plan_sha256, "rubric_lock_sha256": rubric_hash},
    )
    args.resume = args.output.exists() or args.output.with_suffix(".jsonl.manifest.json").exists()
    if output_dir.exists():
        if json.loads((output_dir / "test_run.json").read_text("utf-8")) != build_test_run_record(
            args, plan, integrity
        ):
            raise ValueError("Test run inputs changed; preserve it and choose a new directory")
        verify_test_completion(output_dir, identity)
    return plan, args, integrity, blockers


def prepare_test_recovery(snapshot, args, protocol):
    snapshot = Path(snapshot).resolve()
    receipt = json.loads((snapshot.parent / "snapshot.json").read_text("utf-8"))
    for source, _, record in verified_snapshots(Path(receipt["output"])):
        if source == snapshot:
            recovery = prepare_relocation(source, args)
            if recovery is None or len(recovery["predictions"]) != record["predictions"]:
                raise ValueError("Closed test snapshot has an inconsistent row count")
            checked_protocol(recovery["manifest"]["protocol"], protocol)
            recovery["receipt"]["snapshot_receipt_sha256"] = sha256_file(
                snapshot.parent / "snapshot.json")
            return recovery
    raise ValueError("Recovery requires a verified closed test snapshot")


def run_test(environment_path, plan_path, plan_sha256, identity, output_dir, *, adapter=None,
             rubric=None, inspect_only=False, runner_factory=None, recover_from=None):
    options = dict(adapter=adapter, rubric=rubric)
    plan, args, integrity, blockers = prepare_test(
        environment_path, plan_path, plan_sha256, identity, output_dir, **options)
    recovery = (prepare_test_recovery(recover_from, args, plan["inference_protocol"])
                if recover_from is not None else None)
    report = {"identity": identity, "questions": 1968, "image_integrity": integrity,
              "status": "blocked" if blockers else "inputs_verified",
              "blockers": blockers, "test_predictions_generated": False,
              "runtime_verified_in_this_step": False}
    if recovery is not None:
        report["recovery"] = recovery["receipt"]
    if inspect_only:
        return report
    if blockers:
        raise ValueError(" ".join(blockers))
    output_dir = args.output.parent
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir) + ".lock", timeout=0):
        plan, args, integrity, blockers = prepare_test(
            environment_path, plan_path, plan_sha256, identity, output_dir, **options)
        report["image_integrity"] = integrity
        if blockers:
            raise ValueError(" ".join(blockers))
        record = build_test_run_record(args, plan, integrity)
        receipt = output_dir / "test_run.json"
        if receipt.exists():
            if json.loads(receipt.read_text("utf-8")) != record:
                raise ValueError("Test run inputs changed; preserve it and choose a new directory")
        else:
            output_dir.mkdir(exist_ok=False)
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
        if recovery is not None:
            return {**report, "status": "recovered", "recovery": write_relocation(recovery),
                    "next_step": "Run again without --recover-from to resume inference."}
        records = select_records(read_jsonl(args.annotations), "test", 0, args.seed)
        require_safe_snapshot_resume(args.output, records)
        manifest_path = args.output.with_suffix(".jsonl.manifest.json")
        if manifest_path.exists():
            checked_protocol(json.loads(manifest_path.read_text("utf-8"))["protocol"],
                             plan["inference_protocol"])

        def inputs(arguments, rows, hashes):
            result = export_inputs(arguments, rows, hashes)
            if (result["source_files"] != plan["inference_source"]
                    or result["adapter_files"] != plan["runs"][identity]["adapter_files"]):
                raise ValueError("Inference source or adapter changed before export")
            return result

        progress = {"saved": len(read_jsonl(args.output)) if args.output.exists() else 0,
                    "runtime_verified": False}
        saved_before = progress["saved"]

        class PinnedRunner:
            def __init__(self, *a, **kw):
                if runner_factory is None:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("Run test inference on a connected CUDA GPU")
                self.runner = (runner_factory or Qwen25VLRunner)(*a, **kw)

            def protocol_metadata(self, max_new_tokens):
                metadata = self.runner.protocol_metadata(max_new_tokens)
                checked_protocol(metadata, plan["inference_protocol"])
                progress["runtime_verified"] = True
                return metadata

            def generate(self, *a, **kw):
                # The preceding row has been flushed before the next generate call.
                if progress["saved"] and (progress["saved"] == 25 or progress["saved"] % 100 == 0):
                    snapshot_export(args.output, records)
                result = self.runner.generate(*a, **kw)
                progress["saved"] += 1
                return result

        export_predictions(args, runner_factory=PinnedRunner, input_builder=inputs)
        manifest = json.loads(manifest_path.read_text("utf-8"))
        checked_protocol(manifest["protocol"], plan["inference_protocol"])
        saved = read_jsonl(args.output)
        if len(validate_saved_predictions(saved, records, manifest)) != 1968:
            raise ValueError("Test export is incomplete")
        completion = {"identity": identity, "predictions": 1968,
                      "predictions_sha256": sha256_file(args.output),
                      "manifest_sha256": sha256_file(manifest_path),
                      "run_sha256": sha256_file(receipt)}
        closed = output_dir / "completion.json"
        if closed.exists():
            if json.loads(closed.read_text("utf-8")) != completion:
                raise ValueError("Completed test export changed")
        else:
            write_json(closed, completion)
        snapshot_export(args.output, records)
    return {**report, "status": "completed", "verified_predictions": 1968,
            "test_predictions_generated": len(saved) > saved_before,
            "runtime_verified_in_this_step": progress["runtime_verified"],
            "completion": completion}
