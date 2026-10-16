"""Check paired-run compatibility before reporting answer-score differences."""

import argparse
import hashlib
import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.evaluation.answer_metrics import exact_match, token_f1
from vqa_explanation.evaluation.comparison import (
    EXPERIMENT_IDS,
    cluster_bootstrap,
    compare_predictions,
)
from vqa_explanation.evaluation.semantic_metrics import compare_bertscore_reports

FULL_VALIDATION_COUNT = 1407
FULL_TEST_COUNT = 1968


def read_expected_validation_annotations(
    path: Path, *, expected_count: int = FULL_VALIDATION_COUNT, expected_sha256: str | None = None
) -> tuple[dict[str, dict], str]:
    return read_expected_annotations(
        path, split="validation", expected_count=expected_count, expected_sha256=expected_sha256
    )


def read_expected_annotations(
    path: Path, *, split: str, expected_count: int, expected_sha256: str | None = None
) -> tuple[dict[str, dict], str]:
    """Read and validate frozen reference records."""
    if split not in ("validation", "test"):
        raise ValueError("Only validation or test references are allowed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError(f"Expected {split} SHA-256 must be a 64-character hex digest")
        if digest != expected_sha256:
            raise ValueError(f"{split.capitalize()} annotation SHA-256 mismatch: got {digest}")

    rows = list(read_jsonl(path))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} annotations, found {len(rows)}")
    annotations = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Validation annotation row {index} must be an object")
        if row.get("split") != split:
            raise ValueError(f"Annotation row {index} has a non-{split} split")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Validation annotation row {index} has an invalid question_id")
        if qid in annotations:
            raise ValueError(f"Duplicate validation question_id: {qid}")
        for field in ("image_id", "image_path", "question", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Validation annotation row {index} has an invalid {field}")
        explanations = row.get("explanations")
        if not isinstance(explanations, list) or not explanations or any(
            not isinstance(value, str) or not value.strip() for value in explanations
        ):
            raise ValueError(f"Validation annotation row {index} has invalid explanations")
        annotations[qid] = row
    return annotations, digest


def read_expected_validation_ids(
    path: Path, *, expected_count: int = FULL_VALIDATION_COUNT, expected_sha256: str | None = None
) -> tuple[set[str], str]:
    """Compatibility helper for callers that only need the validated ID set."""
    annotations, digest = read_expected_validation_annotations(
        path, expected_count=expected_count, expected_sha256=expected_sha256
    )
    return set(annotations), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-validation-annotations",
        type=Path,
        help="Require both runs to cover all 1,407 IDs in the frozen validation.jsonl",
    )
    parser.add_argument(
        "--expected-validation-sha256",
        help="Optional SHA-256 pin for the validation annotation file",
    )
    parser.add_argument("--expected-annotations", type=Path,
                        help="Frozen full validation/test references; use with --split and hash")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--left-experiment", choices=EXPERIMENT_IDS, default="M0_PRETRAINED")
    parser.add_argument("--right-experiment", choices=EXPERIMENT_IDS, default="M1_SFT")
    parser.add_argument("--checkpoint-comparison", action="store_true",
                        help="Compare two distinct M1 checkpoints on full pinned validation")
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--left-semantic-report",
        type=Path,
        help="Item-level BERTScore report produced from the left predictions",
    )
    parser.add_argument(
        "--right-semantic-report",
        type=Path,
        help="Item-level BERTScore report produced from the right predictions",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    generic = (args.expected_annotations, args.expected_sha256, args.split)
    if any(generic) and not all(generic):
        parser.error("Use --expected-annotations, --expected-sha256 and --split together")
    if any(generic) and (args.expected_validation_annotations or args.expected_validation_sha256):
        parser.error("Do not mix generic and legacy validation flags")
    if (not args.expected_annotations and not args.expected_validation_annotations
            and (args.left_experiment, args.right_experiment) != ("M0_PRETRAINED", "M1_SFT")):
        parser.error("Explicit experiment roles require canonical annotation references")
    if args.expected_validation_sha256 and not args.expected_validation_annotations:
        parser.error("--expected-validation-sha256 requires --expected-validation-annotations")
    if args.expected_validation_annotations and not args.expected_validation_sha256:
        parser.error("Official validation comparisons require --expected-validation-sha256")
    if args.bootstrap_resamples and not (
        args.expected_annotations or args.expected_validation_annotations
    ):
        parser.error("Bootstrap requires frozen canonical annotation references")
    if args.bootstrap_resamples < 0 or args.bootstrap_resamples == 1:
        parser.error("--bootstrap-resamples must be zero (disabled) or at least two")
    if bool(args.left_semantic_report) != bool(args.right_semantic_report):
        parser.error("Use both --left-semantic-report and --right-semantic-report")
    if args.left_semantic_report and not (
        args.expected_annotations or args.expected_validation_annotations
    ):
        parser.error("Semantic comparison requires frozen canonical annotation references")
    try:
        expected_annotations = None
        annotation_digest = None
        if args.expected_validation_annotations:
            expected_annotations, annotation_digest = read_expected_validation_annotations(
                args.expected_validation_annotations,
                expected_sha256=args.expected_validation_sha256,
            )
        if args.expected_annotations:
            expected_annotations, annotation_digest = read_expected_annotations(
                args.expected_annotations, split=args.split,
                expected_count=FULL_TEST_COUNT if args.split == "test" else FULL_VALIDATION_COUNT,
                expected_sha256=args.expected_sha256,
            )
        reference_args = (
            {"expected_annotations": expected_annotations, "expected_split": args.split}
            if args.expected_annotations else
            {"expected_validation_annotations": expected_annotations}
        )
        left = list(read_jsonl(args.left))
        right = list(read_jsonl(args.right))
        report = compare_predictions(
            left,
            right,
            expected_experiments=(args.left_experiment, args.right_experiment),
            checkpoint_comparison=args.checkpoint_comparison,
            **reference_args,
        )
        if expected_annotations is not None:
            report["expected_reference" if args.expected_annotations else "expected_validation"] = {
                "annotations": str(
                    args.expected_annotations or args.expected_validation_annotations
                ),
                "split": args.split or "validation",
                "sha256": annotation_digest,
                "n": len(expected_annotations),
            }
        if args.bootstrap_resamples and report['comparable']:
            a = {row['question_id']: row for row in left}
            b = {row['question_id']: row for row in right}
            ids = sorted(a)
            differences = {
                name: [metric(b[qid]['prediction_answer'], a[qid]['reference_answer'])
                       - metric(a[qid]['prediction_answer'], a[qid]['reference_answer'])
                       for qid in ids]
                for name, metric in (('exact_match', exact_match), ('token_f1', token_f1))
            }
            report['paired_answer_bootstrap'] = cluster_bootstrap(
                differences, [a[qid]['image_id'] for qid in ids],
                seed=args.bootstrap_seed, resamples=args.bootstrap_resamples,
            )
        if args.left_semantic_report and report["comparable"]:
            left_semantic = json.loads(args.left_semantic_report.read_text(encoding="utf-8"))
            right_semantic = json.loads(args.right_semantic_report.read_text(encoding="utf-8"))
            a = {row["question_id"]: row for row in left}
            semantic, semantic_differences, semantic_image_ids = compare_bertscore_reports(
                left_semantic,
                right_semantic,
                left_predictions_sha256=hashlib.sha256(args.left.read_bytes()).hexdigest(),
                right_predictions_sha256=hashlib.sha256(args.right.read_bytes()).hexdigest(),
                expected_split=args.split or "validation",
                expected_question_images={qid: row["image_id"] for qid, row in a.items()},
            )
            report["explanation_bertscore"] = semantic
            if args.bootstrap_resamples:
                report["paired_explanation_bertscore_bootstrap"] = cluster_bootstrap(
                    {"bertscore_f1": semantic_differences},
                    semantic_image_ids,
                    seed=args.bootstrap_seed,
                    resamples=args.bootstrap_resamples,
                )
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        parser.error(str(exc))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        parser.error(f"Could not write comparison report: {exc}")
    print(f"Comparable: {report['comparable']}; issues: {len(report['issues'])}")
    print(f"Report: {args.output}")
    raise SystemExit(0 if report["comparable"] else 2)


if __name__ == "__main__":
    main()
