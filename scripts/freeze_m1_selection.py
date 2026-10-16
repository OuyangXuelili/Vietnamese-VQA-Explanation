"""Write a verified, immutable M1 selection record without loading a model."""

import argparse
import json
from pathlib import Path

from vqa_explanation.training.preference_data import freeze_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "adapter", "model-config", "annotations", "epoch-result", "trainer-state",
        "loss-selection", "validation-predictions", "validation-manifest",
        "validation-annotations", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--human-review", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_selection(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
