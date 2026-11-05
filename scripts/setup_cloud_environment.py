"""Install a named cloud dependency profile into an isolated interpreter."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("preference", "semantic", "demo"), required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vqa_explanation.utils.cloud_environment import prepare_environment

    python = prepare_environment(root, args.environment_dir, profile=args.profile)
    print(f"{args.profile} Python ready: {python}", flush=True)


if __name__ == "__main__":
    main()
