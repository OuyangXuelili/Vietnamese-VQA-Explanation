"""Single-GPU QLoRA smoke training and validation."""

import argparse
import hashlib
import importlib.metadata
import math
import subprocess
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.training.sft_data import VQACollator, select_training_records


def validate_smoke_completion(
    global_step: int, target_step: int, training_loss: float, weight_delta: float
) -> None:
    if type(global_step) is not int or global_step != target_step or target_step <= 0:
        raise RuntimeError(f"SFT smoke stopped at step {global_step}, not target {target_step}")
    if not math.isfinite(training_loss):
        raise RuntimeError("Non-finite SFT smoke training loss")
    if not math.isfinite(weight_delta) or weight_delta <= 0:
        raise RuntimeError("LoRA weights did not update")


def supports_native_bf16(cuda) -> bool:
    """Check native BF16 support, excluding emulation."""
    if not cuda.is_available() or cuda.get_device_capability(0)[0] < 8:
        return False
    try:
        return cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older Torch releases do not expose including_emulation.
        return cuda.is_bf16_supported()


def load_qlora_model(cfg: dict, model_cfg: dict):
    """Use the same processor, quantization and language-only LoRA for pilot and SFT."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This entry point requires a CUDA GPU")
    set_seed(cfg["seed"])
    bf16 = supports_native_bf16(torch.cuda)
    dtype = torch.bfloat16 if bf16 else torch.float16
    model_id = model_cfg["model"]["id"]
    revision = model_cfg["model"]["revision"]
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    peft = model_cfg["peft"]
    targets = [
        name
        for name, _ in model.named_modules()
        if "visual" not in name and name.rsplit(".", 1)[-1] in peft["target_modules"]
    ]
    if not targets:
        raise RuntimeError("No language attention projections found for LoRA")
    model = get_peft_model(
        model,
        LoraConfig(
            r=peft["rank"],
            lora_alpha=peft["alpha"],
            lora_dropout=peft["dropout"],
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    return processor, model, bf16, dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/sft_smoke.yaml"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/qwen25_vl_3b.yaml"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--all-train-records",
        action="store_true",
        help="Validate every train record instead of the smoke subset; preflight only",
    )
    args = parser.parse_args()
    if args.all_train_records and not args.preflight_only:
        parser.error("--all-train-records requires --preflight-only")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    records = select_training_records(
        list(read_jsonl(args.annotations)),
        0 if args.all_train_records else cfg["limit"],
        cfg["seed"],
    )
    root = args.image_root.resolve()
    for record in records:
        path = (root / record["image_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {record['image_path']}")
    if args.preflight_only:
        print(f"Data preflight passed: {len(records)} train questions; images present")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")

    import torch
    from transformers import Trainer, TrainingArguments

    processor, model, bf16, dtype = load_qlora_model(cfg, model_cfg)
    collator = VQACollator(processor, root, cfg["max_length"])
    for record in records:
        collator([record])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata = {
        "experiment_id": "M1_SFT_SMOKE",
        "config": cfg,
        "model_config": model_cfg,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "question_ids": [r["question_id"] for r in records],
        "reference_policy": "first explanation per training question",
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": str(dtype),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
        },
        "status": "started",
        "source_sha256": {
            str(path.relative_to(Path(__file__).parents[3])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
    }
    write_json(args.output_dir / "run_metadata.json", metadata)

    class FiniteTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                **kwargs,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SFT loss")
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg["seed"],
        optim="adamw_torch",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )
    trainer = FiniteTrainer(
        model=model,
        args=training_args,
        train_dataset=records,
        data_collator=collator,
    )
    tracked = next(p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad)
    before = tracked.detach().float().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    delta = (tracked.detach().float().cpu() - before).abs().max().item()
    validate_smoke_completion(
        trainer.state.global_step, cfg["max_steps"], result.training_loss, delta
    )
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    processor.save_pretrained(str(adapter))
    trainer.save_state()
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() and (adapter / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("SFT smoke adapter files are missing")
    write_json(
        args.output_dir / "smoke_result.json",
        {
            "status": "passed",
            "global_step": trainer.state.global_step,
            "target_global_step": cfg["max_steps"],
            "metrics": result.metrics,
            "max_lora_weight_change": delta,
            "adapter": str(adapter),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "log_history": trainer.state.log_history,
            "note": "Training smoke only; adapter reload and validation inference still required",
        },
    )
    metadata["status"] = "training_smoke_passed"
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(f"SFT smoke passed; adapter: {adapter}")
