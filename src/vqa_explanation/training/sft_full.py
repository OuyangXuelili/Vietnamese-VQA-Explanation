"""Full M1 SFT: fixed data protocol, validation selection and checkpoint resume."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from itertools import chain
from pathlib import Path

import yaml

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.training.image_integrity import verify_record_images
from vqa_explanation.training.sft import load_qlora_model, supports_native_bf16
from vqa_explanation.training.sft_data import VQACollator, select_training_records
from vqa_explanation.utils.run_provenance import sha256_file

CONFIG_KEYS = {
    "stage",
    "experiment_id",
    "seed",
    "expected_train_count",
    "expected_validation_count",
    "expected_test_count",
    "eval_sample_size",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "gradient_accumulation_steps",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "min_pixels",
    "max_pixels",
    "max_length",
}
TESTED_TRANSFORMERS_VERSION = "4.57.6"
RUNTIME_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise ValueError(f"Full SFT config must contain exactly {sorted(CONFIG_KEYS)}")
    if cfg["stage"] != "sft" or cfg["experiment_id"] not in {"M1_SFT", "M1_SFT_PILOT"}:
        raise ValueError("SFT must identify M1_SFT or M1_SFT_PILOT")
    positive_ints = CONFIG_KEYS - {
        "stage",
        "experiment_id",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
    }
    for key in positive_ints:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "num_train_epochs"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a positive finite number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    steps = cfg["max_steps"]
    if (
        type(steps) is not int
        or (cfg["experiment_id"] == "M1_SFT" and steps != -1)
        or (cfg["experiment_id"] == "M1_SFT_PILOT" and steps <= 0)
    ):
        raise ValueError("max_steps must be -1 for full M1 or positive for its pilot")
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    if cfg["eval_sample_size"] > cfg["expected_validation_count"]:
        raise ValueError("eval_sample_size exceeds validation set")
    if cfg["save_steps"] % cfg["eval_steps"]:
        raise ValueError("save_steps must be a multiple of eval_steps for best-model selection")
    if cfg["save_total_limit"] < 2:
        raise ValueError("Retain at least two checkpoints for interrupted cloud runs")


def _ids_sha256(records: list[dict]) -> str:
    ids = [row["question_id"] for row in records]
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_data(
    records: list[dict], image_root: Path, cfg: dict
) -> tuple[list[dict], list[dict], dict]:
    """Select train records and a fixed validation subset."""
    validate_config(cfg)
    audit = audit_records(records)
    counts = audit["counts"]
    expected = {
        "train": cfg["expected_train_count"],
        "validation": cfg["expected_validation_count"],
        "test": cfg["expected_test_count"],
    }
    for split, size in expected.items():
        if counts.get(f"split_{split}", 0) != size:
            raise ValueError(
                f"Expected {size} {split} records; got {counts.get(f'split_{split}', 0)}"
            )
    if audit["errors"] or counts.get("duplicate_question_ids", 0):
        raise ValueError("Invalid or duplicate questions in annotation manifest")
    for row in records:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Noncanonical split label in annotation manifest: {row['split']}")
    if any(audit["image_overlap_by_split"].values()):
        raise ValueError("Image overlap between train/validation/test")
    root = image_root.resolve()
    path_splits: dict[Path, str] = {}
    for row in records:
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Unsafe image path: {row['image_path']}")
        previous_split = path_splits.setdefault(image, row["split"])
        if previous_split != row["split"]:
            raise ValueError(
                f"Image path overlaps {previous_split} and {row['split']}: {row['image_path']}"
            )
    train = select_training_records(records, 0, cfg["seed"])
    validation = [row for row in records if row["split"] == "validation"]
    random.Random(cfg["seed"]).shuffle(validation)
    validation = validation[: cfg["eval_sample_size"]]
    for row in chain(train, validation):
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(
                f"Missing or unsafe image for {row['question_id']}: {row['image_path']}"
            )
    plan = {
        "experiment_id": cfg["experiment_id"],
        "protocol": "image_disjoint_v1",
        "train_count": len(train),
        "validation_count": counts["split_validation"],
        "test_count": counts["split_test"],
        "checkpoint_validation_count": len(validation),
        "train_question_ids_sha256": _ids_sha256(train),
        "checkpoint_validation_question_ids": [row["question_id"] for row in validation],
        "checkpoint_validation_question_ids_sha256": _ids_sha256(validation),
        "reference_policy": "first explanation per question for train and validation loss",
        "split_overlap": audit["image_overlap_by_split"],
    }
    return train, validation, plan


def validate_checkpoint(path: Path) -> Path:
    checkpoint = path.resolve()
    if (
        not checkpoint.is_dir()
        or not checkpoint.name.startswith("checkpoint-")
        or not checkpoint.name[11:].isdigit()
    ):
        raise ValueError("Resume source must be a Trainer checkpoint directory")
    expected_step = int(checkpoint.name[11:])
    if expected_step <= 0:
        raise ValueError("Checkpoint step must be positive")
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        file = checkpoint / required
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Incomplete resume checkpoint: missing {required}")
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid trainer_state.json in resume checkpoint") from error
    if not isinstance(state, dict) or type(state.get("global_step")) is not int:
        raise ValueError("Checkpoint trainer_state.json needs an integer global_step")
    if state["global_step"] != expected_step:
        raise ValueError("Checkpoint global_step does not match its directory name")
    adapter_config = checkpoint / "adapter_config.json"
    if (
        not adapter_config.is_file()
        or adapter_config.stat().st_size == 0
        or not any(
            (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    ):
        raise ValueError("Incomplete resume checkpoint: missing adapter config or weights")
    return checkpoint


def validate_resume_runtime(checkpoint: Path | None, versions: dict[str, str]) -> None:
    """Require matching dependency versions for resume."""
    if versions.get("transformers") != TESTED_TRANSFORMERS_VERSION:
        raise ValueError(
            f"M1 requires tested Transformers {TESTED_TRANSFORMERS_VERSION}; "
            f"got {versions.get('transformers')}"
        )
    if checkpoint is None:
        return
    metadata_path = checkpoint.parent / "run_metadata.json"
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read previous run_metadata.json for resume") from error
    if not isinstance(previous, dict) or previous.get("versions") != versions:
        raise ValueError("Resume runtime versions differ from the checkpoint run")


def validate_resume_plan(checkpoint: Path | None, plan: dict) -> None:
    """Check data and model settings when calling the trainer directly."""
    if checkpoint is None:
        return
    path = checkpoint.parent / "data_plan.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read previous data_plan.json for resume") from error
    if previous != plan:
        raise ValueError("Resume data plan differs from the checkpoint run")


def validate_resume_dtype(checkpoint: Path | None, compute_dtype: str) -> None:
    """Require the checkpoint's compute dtype for resume."""
    if checkpoint is None:
        return
    previous = json.loads((checkpoint.parent / "run_metadata.json").read_text("utf-8"))
    if previous.get("compute_dtype") != compute_dtype:
        raise ValueError(
            "Resume compute dtype differs; keep the original runtime or start a new run"
        )


def validate_session_stop_target(target: int | None, resume_step: int, cfg: dict) -> None:
    if target is None:
        return
    if type(target) is not int or target <= resume_step:
        raise ValueError("Session stop must be after the resume checkpoint step")
    if target % cfg["save_steps"] or target % cfg["eval_steps"]:
        raise ValueError("Session stop must align with both save and evaluation steps")
    if cfg["max_steps"] > 0 and target >= cfg["max_steps"]:
        raise ValueError("Session stop must precede the configured final step")


def validate_training_completion(
    global_step: int,
    max_steps: int,
    resume_step: int,
    training_loss: float,
    weight_delta: float | None,
) -> None:
    """Validate completion and progress for this training session."""
    if global_step <= resume_step:
        raise RuntimeError("No new optimizer step was completed in this run")
    if global_step != max_steps:
        raise RuntimeError(f"Training stopped at step {global_step}, before target {max_steps}")
    if not math.isfinite(training_loss):
        raise RuntimeError("Non-finite training loss")
    if resume_step == 0 and (
        weight_delta is None or not math.isfinite(weight_delta) or weight_delta <= 0
    ):
        raise RuntimeError("LoRA weights did not update during SFT")


def training_arguments(cfg: dict, output_dir: Path, bf16: bool):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        save_only_model=False,
        logging_steps=cfg["logging_steps"],
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        optim="adamw_torch",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--image-checksums", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    if args.preflight_only and (args.resume_from_checkpoint or args.stop_after_step):
        parser.error("A preflight cannot resume or pause training")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; use a fresh run directory")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    records = list(read_jsonl(args.annotations))
    train, validation, plan = plan_data(records, args.image_root, cfg)
    plan["image_integrity"] = verify_record_images(
        chain(train, validation), args.image_root, args.image_checksums
    )
    plan["annotations_sha256"] = sha256_file(args.annotations)
    plan["training_config_sha256"] = sha256_file(args.config)
    plan["model_config_sha256"] = sha256_file(args.model_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "data_plan.json", plan)
    print(
        f"M1 data plan: {len(train)} train, {len(validation)} fixed validation; test unused",
        flush=True,
    )
    if args.preflight_only:
        return

    checkpoint = (
        validate_checkpoint(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    resume_step = int(checkpoint.name.removeprefix("checkpoint-")) if checkpoint else 0
    validate_session_stop_target(args.stop_after_step, resume_step, cfg)
    validate_resume_plan(checkpoint, plan)
    import torch
    from transformers import Trainer, TrainerCallback

    versions = {package: importlib.metadata.version(package) for package in RUNTIME_PACKAGES}
    validate_resume_runtime(checkpoint, versions)
    validate_resume_dtype(
        checkpoint, "torch.bfloat16" if supports_native_bf16(torch.cuda) else "torch.float16",
    )
    processor, model, bf16, dtype = load_qlora_model(cfg, model_cfg)
    collator = VQACollator(processor, args.image_root, cfg["max_length"])
    metadata = {
        "experiment_id": cfg["experiment_id"],
        "status": "started",
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": str(dtype),
        "versions": versions,
    }
    write_json(args.output_dir / "run_metadata.json", metadata)

    class FiniteTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SFT loss")
            return (loss, outputs) if return_outputs else loss

    class CheckpointStopCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if args_stop >= state.max_steps:
                raise ValueError("Session stop must precede the Trainer final step")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == args_stop:
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

    args_stop = args.stop_after_step

    trainer = FiniteTrainer(
        model=model,
        args=training_arguments(cfg, args.output_dir, bf16),
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collator,
        callbacks=[CheckpointStopCallback()] if args_stop is not None else None,
    )
    tracked = next(
        (
            parameter
            for name, parameter in model.named_parameters()
            if "lora_B" in name and parameter.requires_grad
        ),
        None,
    )
    if tracked is None:
        raise RuntimeError("No trainable LoRA-B parameter to verify optimizer updates")
    # Measure updates after Trainer restores the checkpoint inside train().
    before = tracked.detach().float().cpu().clone() if checkpoint is None else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    train_seconds = time.perf_counter() - started
    weight_delta = (
        (tracked.detach().float().cpu() - before).abs().max().item() if before is not None else None
    )
    if args_stop is not None:
        if trainer.state.global_step != args_stop:
            raise RuntimeError("Session did not stop at the requested optimizer step")
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Non-finite training loss in paused session")
        if resume_step == 0 and (
            weight_delta is None or not math.isfinite(weight_delta) or weight_delta <= 0
        ):
            raise RuntimeError("LoRA weights did not update in paused session")
        saved = validate_checkpoint(args.output_dir / f"checkpoint-{args_stop}")
        write_json(
            args.output_dir / "session_result.json",
            {
                "status": "paused",
                "experiment_id": cfg["experiment_id"],
                "global_step": trainer.state.global_step,
                "target_global_step": trainer.state.max_steps,
                "resumed_from_global_step": resume_step if checkpoint else None,
                "optimizer_steps_this_run": trainer.state.global_step - resume_step,
                "checkpoint": str(saved),
                "train_seconds": train_seconds,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "max_lora_weight_change": weight_delta,
                "train_metrics": result.metrics,
                "note": "Session checkpoint only; not a completed M1 or pilot result",
            },
        )
        metadata["status"] = "training_paused"
        write_json(args.output_dir / "run_metadata.json", metadata)
        print(f"M1 session paused at checkpoint: {saved}", flush=True)
        return
    validate_training_completion(
        trainer.state.global_step,
        trainer.state.max_steps,
        resume_step,
        result.training_loss,
        weight_delta,
    )
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("No validation-selected checkpoint was recorded")
    validation_metrics = trainer.evaluate()
    if not math.isfinite(validation_metrics["eval_loss"]):
        raise RuntimeError("Non-finite validation loss")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    processor.save_pretrained(str(adapter))
    trainer.save_state()
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("Saved adapter files are missing")
    write_json(
        args.output_dir / "training_result.json",
        {
            "status": "passed",
            "experiment_id": cfg["experiment_id"],
            "global_step": trainer.state.global_step,
            "target_global_step": trainer.state.max_steps,
            "resumed_from_global_step": resume_step if checkpoint else None,
            "optimizer_steps_this_run": trainer.state.global_step - resume_step,
            "train_seconds": train_seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "max_lora_weight_change": weight_delta,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "final_validation": validation_metrics,
            "train_metrics": result.metrics,
            "adapter": str(adapter),
            "note": (
                "Optimizer pilot only; not final M1"
                if cfg["max_steps"] > 0
                else "Selected by validation loss; answer/explanation inference is separate"
            ),
        },
    )
    metadata["status"] = "training_passed"
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(f"M1 training passed; selected adapter: {adapter}", flush=True)


if __name__ == "__main__":
    main()
