"""Build the fixed 100-image M1 validation review from saved predictions."""

import argparse
from pathlib import Path

from vqa_explanation.evaluation.review import prepare_validation_review
from vqa_explanation.utils.environment import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    environment = load_environment(args.pop("environment"))
    try:
        print(prepare_validation_review(image_root=environment["image_root"], **args))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
