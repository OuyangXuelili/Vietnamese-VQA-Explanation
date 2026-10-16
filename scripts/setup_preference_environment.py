"""Prepare an isolated interpreter before loading the training dependencies."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vqa_explanation.utils.cloud_environment import prepare_environment

    python = prepare_environment(root, args.environment_dir)
    print(f"Training Python ready: {python}", flush=True)


if __name__ == "__main__":
    main()
