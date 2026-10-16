"""Measure the final M1 checkpoint with its pinned training loss protocol."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_completed_run(execution):
    budget = execution.get("budget", {})
    if (execution.get("stage") != "sft-full" or execution.get("status") != "passed"
            or type(execution.get("returncode")) is not int or execution["returncode"] != 0
            or budget.get("protocol_id") != "vqa_nle_m1_epoch1_v1"
            or budget.get("completion_mode") != "full_epoch"
            or budget.get("stop_after_step") != 3683):
        raise ValueError("Wait for a verified, completed M1 epoch; do not evaluate an active run")


def pinned_training(root):
    if any(name == "vqa_explanation" or name.startswith("vqa_explanation.")
           for name in sys.modules):
        raise ValueError("Run this CLI in a fresh process to load the pinned training source")
    sys.path.insert(0, str(root / "src"))
    from vqa_explanation.training import sft, sft_full
    from vqa_explanation.training.image_integrity import verify_record_images
    from vqa_explanation.utils.environment import load_environment

    return sft, sft_full, verify_record_images, load_environment


def prepare(args):
    root = args.source.resolve(strict=True)
    run = args.run.resolve(strict=True)
    execution = read_json(run / "execution.json")
    require_completed_run(execution)
    checks_path = Path(__file__).with_name("resume_pinned_m1.py")
    checks = load_module("epoch_checkpoint_checks", checks_path)
    checks.verify_source(root, execution["provenance"]["source_files"])
    sft, sft_full, verify_record_images, load_environment = pinned_training(root)
    paths = load_environment(args.environment, root)
    config_path = root / "configs/training/sft.yaml"
    model_path = root / "configs/model/qwen25_vl_3b.yaml"
    files = {"environment": args.environment.resolve(strict=True),
             "annotations": paths["annotations"], "training_config": config_path,
             "model_config": model_path, "image_checksums": paths["image_checksums"]}
    if {name: sha256(path) for name, path in files.items()} != (
        execution["provenance"]["input_sha256"]
    ):
        raise ValueError("Inputs differ from the completed M1 run")
    versions = {name: importlib.metadata.version(name) for name in sft_full.RUNTIME_PACKAGES}
    training = run / "training"
    final = training / "checkpoint-3683"
    sft_full.validate_resume_runtime(final, versions)
    completion = checks.verify_epoch_completion(run, 3683)
    integrity = checks.verify_checkpoint_payloads(final)
    import yaml

    cfg = yaml.safe_load(config_path.read_text("utf-8"))
    model_cfg = yaml.safe_load(model_path.read_text("utf-8"))
    records = list(sft_full.read_jsonl(paths["annotations"]))
    _, validation, plan = sft_full.plan_data(records, paths["image_root"], cfg)
    plan.update(annotations_sha256=sha256(paths["annotations"]),
                training_config_sha256=sha256(config_path), model_config_sha256=sha256(model_path))
    saved_plan = read_json(training / "data_plan.json")
    if {key: value for key, value in saved_plan.items() if key != "image_integrity"} != plan:
        raise ValueError("The fixed validation subset or training protocol changed")
    image_integrity = verify_record_images(
        validation, paths["image_root"], paths["image_checksums"],
    )
    evidence_files = {
        **{str(path): sha256(path) for path in files.values()},
        **{str(training / name): sha256(training / name) for name in (
            "training_result.json", "trainer_state.json", "run_metadata.json", "data_plan.json",
        )},
        str(run / "execution.json"): sha256(run / "execution.json"),
        str(final / "adapter_model.safetensors"): integrity["file_sha256"][
            "adapter_model.safetensors"],
        str(training / "adapter/adapter_model.safetensors"): completion[
            "selected_adapter_sha256"],
    }
    return {"source": root, "run": run, "checkpoint": final, "paths": paths,
            "cfg": cfg, "model_cfg": model_cfg, "validation": validation,
            "sft": sft, "sft_full": sft_full, "completion": completion, "checks": checks,
            "evidence": {"file_sha256": evidence_files, "versions": versions,
                         "source_files": execution["provenance"]["source_files"],
                         "validation_question_ids": plan["checkpoint_validation_question_ids"],
                         "validation_question_ids_sha256": plan[
                             "checkpoint_validation_question_ids_sha256"],
                         "validation_image_integrity": image_integrity,
                         "compute_dtype": integrity["compute_dtype"],
                         "evaluator_sha256": sha256(Path(__file__)),
                         "checkpoint_checker_sha256": sha256(checks_path)}}


def verify_weights(expected, actual):
    import torch

    if not expected or expected.keys() != actual.keys() or any(
        not bool(torch.isfinite(value).all())
        or not torch.equal(value.cpu(), actual[name].detach().cpu())
        for name, value in expected.items()
    ):
        raise ValueError("Loaded adapter weights do not match the final checkpoint")


def evaluate_final(prepared, output):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Trainer

    processor, model, bf16, dtype = prepared["sft"].load_qlora_model(
        prepared["cfg"], prepared["model_cfg"],
    )
    if str(dtype) != prepared["evidence"]["compute_dtype"]:
        raise ValueError("Evaluation compute dtype differs from M1 training")
    weights = load_file(str(prepared["checkpoint"] / "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(model, weights, adapter_name="default")
    verify_weights(weights, get_peft_model_state_dict(model, adapter_name="default"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    class FiniteTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite final-checkpoint validation loss")
            return (loss, outputs) if return_outputs else loss

    full = prepared["sft_full"]
    trainer = FiniteTrainer(
        model=model, args=full.training_arguments(prepared["cfg"], output, bf16),
        eval_dataset=prepared["validation"],
        data_collator=full.VQACollator(
            processor, prepared["paths"]["image_root"], prepared["cfg"]["max_length"],
        ),
    )
    metrics = trainer.evaluate()
    verify_weights(weights, get_peft_model_state_dict(model, adapter_name="default"))
    return metrics


def select_checkpoint(completion, final_loss, run):
    best_loss = completion["selected_eval_loss"]
    for loss in (best_loss, final_loss):
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError("Checkpoint selection requires finite validation losses")
    final_wins = final_loss < best_loss
    return {"selected_step": 3683 if final_wins else completion["selected_step"],
            "selected_checkpoint": str(run / "training/checkpoint-3683") if final_wins else (
                completion["selected_checkpoint"]),
            "selected_adapter": str(run / "training/checkpoint-3683") if final_wins else (
                str(run / "training/adapter")),
            "selected_validation_loss": min(final_loss, best_loss),
            "trainer_selected_loss": best_loss, "final_checkpoint_loss": final_loss,
            "selection_rule": "lower fixed-validation loss; keep Trainer selection on a tie"}


def run(args):
    output = args.output_dir.resolve()
    if output.is_relative_to(args.run.resolve()) or output.is_relative_to(args.source.resolve()):
        raise ValueError("Keep evaluation output outside the finished run and pinned source")
    prepared = prepare(args)
    if args.check_only:
        print(json.dumps(prepared["evidence"], ensure_ascii=False, indent=2))
        print("Post-epoch inputs verified; no model loaded or output written")
        return
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "run": str(prepared["run"]),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": prepared["evidence"], "validation_count": len(prepared["validation"]),
              "test_predictions_generated": 0}
    result_path = output / "selection.json"
    write_report(result_path, report)
    try:
        metrics = evaluate_final(prepared, output)
        for path, digest in prepared["evidence"]["file_sha256"].items():
            if sha256(Path(path)) != digest:
                raise ValueError("M1 inputs changed during evaluation; do not use this selection")
        prepared["checks"].verify_source(
            prepared["source"], prepared["evidence"]["source_files"],
        )
        report.update(select_checkpoint(prepared["completion"], metrics["eval_loss"],
                                        prepared["run"]))
        report.update(
            status="passed", final_checkpoint_metrics=metrics,
            selected_adapter_sha256=sha256(
                Path(report["selected_adapter"]) / "adapter_model.safetensors"),
            reference_policy="first explanation per question; assistant-only loss",
        )
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(result_path, report)
    print(f"M1 selection saved: {result_path}")


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
