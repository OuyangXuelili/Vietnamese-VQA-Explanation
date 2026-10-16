"""Export M1 candidates from a frozen train-only pool; do not assign preference labels."""

import argparse
import json
from pathlib import Path

from scripts.run_m0 import DEFAULT_MODEL, DEFAULT_REVISION
from vqa_explanation.preference.candidates import generate_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--m1-selection', type=Path)
    parser.add_argument('--m1-selection-sha256')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-id', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--min-pixels', type=int, default=200704)
    parser.add_argument('--max-pixels', type=int, default=200704)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if bool(args.m1_selection) != bool(args.m1_selection_sha256):
        parser.error('--m1-selection and --m1-selection-sha256 must be used together')
    print(json.dumps(generate_candidates(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
