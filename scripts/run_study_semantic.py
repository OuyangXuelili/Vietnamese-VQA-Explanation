"""Compute or resume four-arm BERTScore from frozen validation exports."""

import argparse
import json
import os
from pathlib import Path

from vqa_explanation.evaluation.study_semantic import run_semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    os.environ["MPLBACKEND"] = "Agg"
    result = run_semantic(args.results, args.annotations, args.output_dir, device=args.device,
                          batch_size=args.batch_size, cache_dir=args.cache_dir,
                          inspect_only=args.inspect)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
