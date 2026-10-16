"""Restore an immutable preference dataset and its metadata checksums."""

import argparse
from pathlib import Path

from vqa_explanation.training.preference_launch import restore_preference_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(restore_preference_bundle(args.archive, args.sha256, args.destination))


if __name__ == "__main__":
    main()
