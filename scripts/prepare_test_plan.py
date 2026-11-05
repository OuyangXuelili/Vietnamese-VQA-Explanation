"""Verify final-test identities from existing main-study evidence; do not run inference."""

import argparse
import json
from pathlib import Path

from vqa_explanation.evaluation.final_plan import build_final_plan, save_final_plan
from vqa_explanation.utils.environment import load_environment


def by_seed(values):
    result = {}
    for value in values:
        key, path = value.split("=", 1)
        seed = int(key)
        if seed in result:
            raise ValueError(f"Duplicate seed: {seed}")
        result[seed] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", required=True, help="SEED=FOLDER")
    parser.add_argument("--study", action="append", required=True, help="SEED=YAML")
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    environment = load_environment(args.environment)
    result = build_final_plan(by_seed(args.evidence), by_seed(args.study),
                              environment["annotations"], args.sample, args.calibration)
    if args.output_dir is not None:
        save_final_plan(result, args.output_dir)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"sources", "inference_protocol", "inference_source", "runs"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
