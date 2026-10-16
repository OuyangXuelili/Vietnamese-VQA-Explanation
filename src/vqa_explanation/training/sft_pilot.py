"""Measure one GPU forward/backward pass without an optimizer update."""

import argparse
import importlib.metadata
import math
import time
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.training.sft import load_qlora_model
from vqa_explanation.training.sft_data import VQACollator, select_training_records
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def pilot_record(records: list[dict], question_id: str, seed: int) -> dict:
    """Validate the split before selecting the requested train question."""
    train = select_training_records(records, 0, seed)
    matches = [record for record in train if record["question_id"] == question_id]
    if len(matches) != 1:
        raise ValueError(f"Pilot question is not a unique train record: {question_id}")
    return matches[0]


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"seed", "question_id", "min_pixels", "max_pixels", "max_length"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"Pilot config must have exactly {sorted(required)}")
    if not isinstance(cfg["question_id"], str) or not cfg["question_id"]:
        raise ValueError("question_id must be a nonempty string")
    numeric = required - {"question_id"}
    if any(type(cfg[key]) is not int or cfg[key] <= 0 for key in numeric):
        raise ValueError("Pilot numeric config values must be positive integers")
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    return cfg


def pilot(
    annotations: Path,
    image_root: Path,
    output: Path,
    training_config: Path,
    model_config: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite GPU pilot: {output}")
    cfg = load_config(training_config)
    model_cfg = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    if not annotations.is_file() or not image_root.is_dir():
        raise FileNotFoundError("Annotations or image root missing")
    records = list(read_jsonl(annotations))
    record = pilot_record(records, cfg["question_id"], cfg["seed"])

    import torch

    processor, model, bf16, dtype = load_qlora_model(cfg, model_cfg)
    collator = VQACollator(processor, image_root, max_length=cfg["max_length"])
    batch = {key: value.to("cuda:0") for key, value in collator([record]).items()}
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started_at = utc_now()
    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    with torch.autocast("cuda", dtype=dtype):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite pilot loss")
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    grads = [
        param.grad.detach().abs().max()
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad and param.grad is not None
    ]
    if not grads:
        raise RuntimeError("No LoRA gradients in pilot")
    gradient_scale = scaler.get_scale()
    max_gradient = torch.stack(grads).max().item() / gradient_scale
    if not math.isfinite(max_gradient) or max_gradient <= 0:
        raise RuntimeError("LoRA gradients are non-finite or zero")
    gib = 1024**3
    report = {
        "status": "passed",
        "scope": "one forward/backward batch; no optimizer step, checkpoint or validation",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "question_id": record["question_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "annotations_sha256": sha256_file(annotations),
        "training_config_sha256": sha256_file(training_config),
        "model_config_sha256": sha256_file(model_config),
        "model_id": model_cfg["model"]["id"],
        "model_revision": model_cfg["model"]["revision"],
        "min_pixels": cfg["min_pixels"],
        "max_pixels": cfg["max_pixels"],
        "max_length": cfg["max_length"],
        "tokens": int(batch["input_ids"].shape[1]),
        "image_tokens": int(
            (batch["input_ids"] == processor.tokenizer.get_vocab()["<|image_pad|>"]).sum()
        ),
        "supervised_tokens": int((batch["labels"] != -100).sum()),
        "loss": float(loss.detach().float().item()),
        "max_lora_gradient": max_gradient,
        "gradient_scale": gradient_scale,
        "forward_backward_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / gib,
        "baseline_allocated_gib": baseline_allocated / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
        "compute_dtype": str(dtype),
        "bf16": bf16,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
        },
    }
    write_json(output, report)
    print(f"GPU pilot passed: {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/sft_batch_pilot.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen25_vl_3b.yaml")
    )
    args = parser.parse_args()
    pilot(args.annotations, args.image_root, args.output, args.config, args.model_config)


if __name__ == "__main__":
    main()
