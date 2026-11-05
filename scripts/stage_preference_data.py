"""Package or restore the image subset needed by preference training."""

import argparse
import json
from pathlib import Path

from vqa_explanation.data.image_subset import package_image_subset, restore_image_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("package")
    build.add_argument("--environment", type=Path,
                       default=Path("configs/environments/local_full.yaml"))
    build.add_argument("--preferences", type=Path, required=True)
    build.add_argument("--regression-config", type=Path,
                       default=Path("configs/evaluation/numerical_stability.yaml"))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--include-validation", action="store_true")
    restore = commands.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--sha256", required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        result = package_image_subset(args.environment, args.preferences,
                                      args.regression_config, args.output,
                                      include_validation=args.include_validation)
    else:
        result = restore_image_subset(args.archive, args.sha256, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
