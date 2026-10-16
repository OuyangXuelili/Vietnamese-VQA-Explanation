"""Evaluate JSONL predictions produced by the VQA inference scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

from vqa_explanation.evaluation.answer_metrics import exact_match, normalize_answer, token_f1
from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1
from vqa_explanation.evaluation.semantic_metrics import evaluate_bertscore


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Read JSONL predictions and reject malformed rows."""
    return read_prediction_snapshot(path)[0]


def read_prediction_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Bind parsed rows to the exact bytes used for evaluation."""
    payload = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"{path}: invalid JSON on line {line_number}: {error.msg}"
            raise ValueError(message) from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must contain a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records, hashlib.sha256(payload).hexdigest()


def evaluate_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate answer, lexical explanation, format, and latency metrics."""
    if not records:
        raise ValueError("No prediction records to evaluate")

    question_ids = [record.get("question_id") for record in records]
    if any(qid is not None for qid in question_ids):
        if any(not isinstance(qid, str) or not qid.strip() for qid in question_ids):
            raise ValueError("question_id must be a non-empty string on every record")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Duplicate question_id in prediction records")

    answer_scores: list[float] = []
    f1_scores: list[float] = []
    explanation_scores: list[float] = []
    format_values: list[float] = []
    latencies: list[float] = []

    explanation_fields_present = [
        "prediction_explanation" in record or "reference_explanations" in record
        for record in records
    ]
    if any(explanation_fields_present) and not all(explanation_fields_present):
        raise ValueError("Explanation fields must be present on every prediction record")
    score_explanations = all(explanation_fields_present)
    format_fields_present = ["format_valid" in record for record in records]
    if any(format_fields_present) and not all(format_fields_present):
        raise ValueError("format_valid must be present on every prediction record or none")

    for index, record in enumerate(records, start=1):
        for field in ("prediction_answer", "reference_answer"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Record {index} must contain string field {field!r}")
        if not normalize_answer(record["reference_answer"]):
            raise ValueError(f"Record {index} has an empty reference_answer after normalization")
        answer_scores.append(exact_match(record["prediction_answer"], record["reference_answer"]))
        f1_scores.append(token_f1(record["prediction_answer"], record["reference_answer"]))

        if score_explanations:
            try:
                explanation_scores.append(
                    rouge_l_f1(
                        record.get("prediction_explanation"), record.get("reference_explanations")
                    )
                )
            except ValueError as error:
                raise ValueError(f"Record {index}: {error}") from error

        if "format_valid" in record:
            if type(record["format_valid"]) is not bool:
                raise ValueError(f"Record {index} format_valid must be a boolean")
            format_values.append(float(record["format_valid"]))
        if record.get("latency_seconds") is not None:
            try:
                latency = float(record["latency_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Record {index} has an invalid latency_seconds value") from error
            if not math.isfinite(latency) or latency < 0:
                raise ValueError(f"Record {index} latency_seconds must be finite and non-negative")
            latencies.append(latency)

    return {
        "n": len(records),
        "exact_match": mean(answer_scores),
        "token_f1": mean(f1_scores),
        "explanation_n": len(explanation_scores),
        "explanation_status": "scored" if score_explanations else "unavailable",
        "explanation_rouge_l_f1": mean(explanation_scores) if explanation_scores else None,
        "format_valid_n": len(format_values),
        "format_rate": mean(format_values) if format_values else None,
        "latency_n": len(latencies),
        "mean_latency_seconds": mean(latencies) if latencies else None,
        "median_latency_seconds": median(latencies) if latencies else None,
    }


def main() -> None:
    # This CLI does not use Colab's interactive plotting backend.
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Input JSONL prediction file")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--output", type=Path, help="Optional path for the JSON metrics report")
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Also compute the pinned Vietnamese BERTScore protocol",
    )
    parser.add_argument(
        "--bertscore-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for BERTScore; auto uses CUDA when available",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="BERTScore batch size; reduce this if GPU memory is limited",
    )
    parser.add_argument(
        "--bertscore-cache",
        type=Path,
        help="Optional Hugging Face cache directory for the pinned evaluator model",
    )
    args = parser.parse_args()

    if args.output and args.output.exists():
        parser.error("Output exists; choose a new report path")

    try:
        records, predictions_sha256 = read_prediction_snapshot(args.predictions)
        if args.split:
            records = [record for record in records if record.get("split") == args.split]
            if not records:
                raise ValueError(f"No prediction records found for split {args.split!r}")
        metrics = evaluate_predictions(records)
        if args.bertscore:
            metrics["explanation_bertscore"] = evaluate_bertscore(
                records,
                batch_size=args.bertscore_batch_size,
                device=args.bertscore_device,
                cache_dir=args.bertscore_cache,
            )
        metrics["predictions_sha256"] = predictions_sha256
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    metrics["split"] = args.split or "all"
    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(output + "\n")
        console_metrics = dict(metrics)
        semantic = metrics.get("explanation_bertscore")
        if isinstance(semantic, dict) and isinstance(semantic.get("items"), list):
            console_metrics["explanation_bertscore"] = {
                key: value for key, value in semantic.items() if key != "items"
            }
            console_metrics["explanation_bertscore"]["items_saved"] = len(semantic["items"])
        print(json.dumps(console_metrics, ensure_ascii=False, indent=2))
    else:
        print(output)


if __name__ == "__main__":
    main()
