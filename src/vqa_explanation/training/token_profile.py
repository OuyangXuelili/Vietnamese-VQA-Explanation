"""Measure Qwen token lengths on a deterministic train sample."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path

import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.training.sft_data import VQACollator, select_training_records
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def nearest_rank(values: list[int], proportion: float) -> int:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def summarize(measurements: list[dict], max_length: int) -> dict:
    if not measurements:
        raise ValueError("No token measurements")
    result = {"samples": len(measurements), "max_length": max_length}
    for field in ("total_tokens", "supervised_tokens", "image_tokens"):
        values = [entry[field] for entry in measurements]
        result[field] = {
            "min": min(values),
            "p50": nearest_rank(values, 0.50),
            "p90": nearest_rank(values, 0.90),
            "p95": nearest_rank(values, 0.95),
            "p99": nearest_rank(values, 0.99),
            "max": max(values),
        }
    too_long = [entry for entry in measurements if entry["total_tokens"] > max_length]
    result["over_max_length"] = len(too_long)
    result["over_max_length_rate"] = len(too_long) / len(measurements)
    result["longest"] = sorted(
        measurements, key=lambda entry: (-entry["total_tokens"], entry["question_id"])
    )[:20]
    return result


def measure_tokens(question_id: str, input_ids, labels, image_id: int) -> dict:
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    targets = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if len(ids) != len(targets) or not ids:
        raise ValueError(f"Invalid token/label alignment: {question_id}")
    image_tokens = sum(token == image_id for token in ids)
    supervised = sum(label != -100 for label in targets)
    if image_tokens == 0 or supervised == 0:
        raise ValueError(f"Missing image or supervised tokens: {question_id}")
    return {
        "question_id": question_id,
        "total_tokens": len(ids),
        "supervised_tokens": supervised,
        "image_tokens": image_tokens,
    }


def profile(
    annotations: Path,
    image_root: Path,
    output: Path,
    training_config: Path,
    model_config: Path,
    validation_count: int = 0,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite token profile: {output}")
    cfg = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    required = {"seed", "sample_size", "min_pixels", "max_pixels", "max_length"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"Readiness config must have exactly {sorted(required)}")
    if any(not isinstance(cfg[key], int) or cfg[key] <= 0 for key in required):
        raise ValueError("Readiness config values must be positive integers")
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    if type(validation_count) is not int or validation_count < 0:
        raise ValueError("validation_count must be a non-negative integer")
    if not annotations.is_file() or not image_root.is_dir():
        raise FileNotFoundError("Annotations or image root missing")

    records = list(read_jsonl(annotations))
    selected = select_training_records(records, cfg["sample_size"], cfg["seed"])
    if len(selected) != cfg["sample_size"]:
        raise ValueError(
            f"Requested {cfg['sample_size']} samples but only {len(selected)} train records exist"
        )
    validation = [record for record in records if record["split"] == "validation"]
    random.Random(cfg["seed"]).shuffle(validation)
    if validation_count > len(validation):
        raise ValueError("validation_count exceeds available validation records")
    validation = validation[:validation_count]
    measured_records = selected + validation

    from transformers import AutoProcessor

    model = model_cfg["model"]
    processor = AutoProcessor.from_pretrained(
        model["id"],
        revision=model["revision"],
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    collator = VQACollator(processor, image_root, max_length=10**9)
    image_id = processor.tokenizer.get_vocab().get("<|image_pad|>")
    if image_id is None:
        raise ValueError("Processor vocabulary lacks <|image_pad|>")
    measurements = []
    started_at = utc_now()
    for index, record in enumerate(measured_records, 1):
        try:
            batch = collator([record])
        except Exception as exc:
            write_json(
                output,
                {
                    "status": "failed",
                    "question_id": record["question_id"],
                    "checked": len(measurements),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        measurements.append(
            measure_tokens(
                record["question_id"], batch["input_ids"][0], batch["labels"][0], image_id
            )
        )
        if index % 50 == 0 or index == len(measured_records):
            print(f"Token profile: {index}/{len(measured_records)}", flush=True)

    selected_ids = [record["question_id"] for record in selected]
    report = {
        "status": "passed",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "annotations_sha256": sha256_file(annotations),
        "training_config_sha256": sha256_file(training_config),
        "model_config_sha256": sha256_file(model_config),
        "model_id": model["id"],
        "model_revision": model["revision"],
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "pillow")
        },
        "seed": cfg["seed"],
        "train_records": sum(record["split"] == "train" for record in records),
        "sample_size": len(selected),
        "validation_sample_size": len(validation),
        "validation_question_ids": [record["question_id"] for record in validation],
        "selected_question_ids_sha256": hashlib.sha256(
            json.dumps(selected_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "min_pixels": cfg["min_pixels"],
        "max_pixels": cfg["max_pixels"],
        "summary": summarize(measurements, cfg["max_length"]),
        "train_summary": summarize(measurements[:len(selected)], cfg["max_length"]),
        "validation_summary": (
            summarize(measurements[len(selected):], cfg["max_length"]) if validation else None
        ),
        "note": (
            "Full train coverage; validation coverage is reported separately; test excluded"
            if len(selected) == sum(record["split"] == "train" for record in records)
            else "Train sample; validation coverage is reported separately; test excluded"
        ) + "; not a GPU memory measurement",
    }
    write_json(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/sft_readiness.yaml"))
    parser.add_argument(
        "--validation-count", type=int, default=0,
        help="Also scan the first N validation records shuffled with the config seed, as in M1",
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen25_vl_3b.yaml")
    )
    args = parser.parse_args()
    profile(
        args.annotations, args.image_root, args.output, args.config, args.model_config,
        args.validation_count,
    )


if __name__ == "__main__":
    main()
