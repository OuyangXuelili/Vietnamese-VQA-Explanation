from __future__ import annotations

import argparse
from pathlib import Path

from filelock import FileLock, Timeout

from vqa_explanation.evaluation.comparison import EXPERIMENT_IDS
from vqa_explanation.inference.export import export_inputs
from vqa_explanation.inference.export import export_predictions as export_predictions_impl
from vqa_explanation.inference.export import read_jsonl as read_jsonl
from vqa_explanation.inference.export import select_records as select_records
from vqa_explanation.inference.export import validate_selected_records as validate_selected_records
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VQA predictions for an M0–M3 model.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue a verified partial export")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--experiment-id", choices=EXPERIMENT_IDS, default="M0_PRETRAINED")
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    return parser.parse_args()


def export_predictions(args) -> None:
    export_predictions_impl(args, runner_factory=Qwen25VLRunner, input_builder=export_inputs)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(args.output) + ".lock", timeout=0):
            export_predictions(args)
    except Timeout as error:
        raise SystemExit(
            "Another process holds the export lock; do not start a second writer"
        ) from error


if __name__ == "__main__":
    main()
