"""Compare legacy and grid-reuse preprocessing on train images; no model/GPU required."""

import argparse
import importlib.metadata
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.training.sft_data import VQACollator, select_training_records
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.run_provenance import sha256_file


def compare(records, reference, optimized, repeats=3):
    if not records or type(repeats) is not int or repeats <= 0:
        raise ValueError("Nonempty records and positive repeats are required")
    timings = {"reference": [], "optimized": []}
    # Equality is checked for every field, including vision tensors and labels.
    for record in records:
        expected, actual = reference([record]), optimized([record])
        if expected.keys() != actual.keys():
            raise AssertionError(f"Changed batch fields: {record['question_id']}")
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for repeat in range(repeats):
        order = [("reference", reference), ("optimized", optimized)]
        if repeat % 2:
            order.reverse()
        for name, collator in order:
            started = time.perf_counter()
            for record in records:
                collator([record])
            timings[name].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "status": "passed",
        "equal_samples": len(records),
        "question_ids": [r["question_id"] for r in records],
        "seconds": timings,
        "median_seconds": medians,
        "preprocessing_speedup": medians["reference"] / medians["optimized"],
        "note": "CPU preprocessing only; not GPU training speed or a completed M1 run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", type=Path, default=Path("configs/environments/local_full.yaml")
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--return-tensors", choices=("pt", "np"), default="pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("samples and repeats must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new report")
    from transformers import AutoProcessor

    root = project_root()
    paths = load_environment(args.environment)
    cfg = yaml.safe_load((root / "configs/training/sft.yaml").read_text("utf-8"))
    model = yaml.safe_load((root / "configs/model/qwen25_vl_3b.yaml").read_text("utf-8"))
    processor = AutoProcessor.from_pretrained(
        model["model"]["id"],
        revision=model["model"]["revision"],
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
        use_fast=False,
    )
    records = select_training_records(
        list(read_jsonl(paths["annotations"])), args.samples, cfg["seed"]
    )
    reference = VQACollator(
        processor,
        paths["image_root"],
        cfg["max_length"],
        reuse_image_grid=False,
        return_tensors=args.return_tensors,
    )
    optimized = VQACollator(
        processor, paths["image_root"], cfg["max_length"], return_tensors=args.return_tensors,
    )
    report = compare(records, reference, optimized, args.repeats)
    report.update(
        model=model["model"],
        annotations_sha256=sha256_file(paths["annotations"]),
        transformers=importlib.metadata.version("transformers"),
        processor_class=type(processor).__name__,
        image_processor_class=type(processor.image_processor).__name__,
        return_tensors=args.return_tensors,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow", "tokenizers")
        },
    )
    write_json(args.output, report)
    print(
        f"Equal tensors: {len(records)} samples; "
        f"CPU preprocessing {report['preprocessing_speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
