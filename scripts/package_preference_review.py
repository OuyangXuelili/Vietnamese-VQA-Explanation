"""Package anchored M1 train candidates and images for offline human review."""

import argparse
import json
from pathlib import Path

from vqa_explanation.preference.bundle import package_preference_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-dir", "pool-dir", "m1-selection", "image-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    for name in ("annotations", "pool-questions", "pool-manifest", "m1-selection"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(package_preference_review(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
