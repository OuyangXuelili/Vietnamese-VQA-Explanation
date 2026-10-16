"""Compare paired VQA predictions with explicit protocol checks."""

from statistics import mean

import numpy as np

from .answer_metrics import exact_match, normalize_answer, token_f1
from .explanation_metrics import rouge_l_f1

# Both runs must use the same inference protocol for an adapter-only comparison.
PROTOCOL_FIELDS = (
    "model_id",
    "model_revision",
    "seed",
    "max_pixels",
    "min_pixels",
    "max_new_tokens",
    "prompt_sha256",
    "generation_config",
    "load_in_4bit",
    "compute_dtype",
)
REFERENCE_FIELDS = (
    "image_id",
    "image_path",
    "image_sha256",
    "question",
    "reference_answer",
    "split",
)
EXPERIMENT_IDS = ("M0_PRETRAINED", "M1_SFT", "M2_SFT_CHOSEN", "M3_DPO")


def cluster_bootstrap(differences, image_ids, *, seed=42, resamples=2000):
    """Question-weighted paired deltas; sample entire image clusters together.

    Call only after checking run compatibility and canonical ID coverage.
    This measures sampling uncertainty, not training-seed variability.
    """
    if type(resamples) is not int or resamples < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Use at least two resamples and a nonnegative integer seed")
    if (not image_ids or any(not isinstance(value, str) or not value for value in image_ids)
            or not differences):
        raise ValueError("Image IDs and metric differences must be nonempty")
    groups = {}
    for index, image_id in enumerate(image_ids):
        groups.setdefault(image_id, []).append(index)
    if len(groups) < 2:
        raise ValueError("At least two image clusters are required")
    names = sorted(differences)
    values = np.asarray([differences[name] for name in names], dtype=float)
    if values.shape != (len(names), len(image_ids)) or not np.isfinite(values).all():
        raise ValueError("Metric differences must be finite and align with the image IDs")
    clusters = np.asarray([
        [len(indices), *values[:, indices].sum(axis=1)]
        for _, indices in sorted(groups.items())
    ])
    rng = np.random.default_rng(seed)
    totals = np.asarray([
        clusters[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
        for _ in range(resamples)
    ])
    scores = totals[:, 1:] / totals[:, :1]
    return {
        'method': 'paired_image_cluster_percentile_bootstrap',
        'cluster_order': 'sorted_image_id', 'weighting': 'per_question',
        'n_questions': len(image_ids), 'n_image_clusters': len(groups),
        'seed': seed, 'resamples': resamples,
        'metrics': {
            name: {
                'delta_right_minus_left': float(values[index].mean()),
                'ci95': np.quantile(scores[:, index], [.025, .975]).tolist(),
            }
            for index, name in enumerate(names)
        },
        'limitations': 'Sampling uncertainty within these runs, not training-seed stability, '
                       'visual grounding, or proof of a causal treatment effect.',
    }


def compare_predictions(
    left: list[dict],
    right: list[dict],
    *,
    expected_validation_ids: set[str] | None = None,
    expected_validation_annotations: dict[str, dict] | None = None,
    expected_annotations: dict[str, dict] | None = None,
    expected_split: str = "validation",
    expected_experiments: tuple[str, str] = ("M0_PRETRAINED", "M1_SFT"),
    checkpoint_comparison: bool = False,
) -> dict:
    issues = []
    if expected_split not in ("validation", "test"):
        raise ValueError("Expected split must be validation or test")
    if type(checkpoint_comparison) is not bool:
        raise ValueError("checkpoint_comparison must be a boolean")
    if checkpoint_comparison and (
        expected_experiments != ("M1_SFT", "M1_SFT") or expected_split != "validation"
    ):
        raise ValueError("Checkpoint comparison is only for M1 on validation")
    if (len(expected_experiments) != 2
            or any(name not in EXPERIMENT_IDS for name in expected_experiments)
            or (expected_experiments[0] == expected_experiments[1]
                and not checkpoint_comparison)):
        raise ValueError("Expected experiments must be two distinct M0-M3 experiment IDs")
    if expected_annotations is not None:
        if expected_validation_annotations is not None or expected_validation_ids is not None:
            raise ValueError("Do not mix generic and legacy validation references")
        expected_validation_annotations = expected_annotations
    elif expected_split != "validation":
        raise ValueError("Test comparison requires expected_annotations")
    if (expected_validation_annotations is None
            and expected_experiments != ("M0_PRETRAINED", "M1_SFT")):
        raise ValueError("Explicit experiment roles require canonical annotation references")
    if expected_validation_annotations is not None:
        for qid, row in expected_validation_annotations.items():
            if (not isinstance(row, dict) or row.get("question_id") != qid
                    or row.get("split") != expected_split):
                raise ValueError("Expected annotation key/question ID or split mismatch")
    if expected_validation_annotations is not None:
        annotation_ids = set(expected_validation_annotations)
        if expected_validation_ids is not None and expected_validation_ids != annotation_ids:
            raise ValueError("Expected validation IDs disagree with annotation records")
        expected_validation_ids = annotation_ids
    if expected_validation_ids is not None and (
        not expected_validation_ids
        or any(not isinstance(qid, str) or not qid for qid in expected_validation_ids)
    ):
        raise ValueError("Expected validation question IDs must be nonempty strings")

    def index_records(records, label):
        indexed = {}
        if not records:
            issues.append(f"{label}: empty predictions")
        protocols = []
        for i, row in enumerate(records):
            if not isinstance(row, dict):
                issues.append(f"{label}[{i}]: prediction must be an object")
                continue
            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid.strip():
                issues.append(f"{label}[{i}]: missing/non-string question_id")
                continue
            if qid in indexed:
                issues.append(f"{label}: duplicate question_id {qid}")
            indexed[qid] = row
            for field in (*PROTOCOL_FIELDS, *REFERENCE_FIELDS, "prediction_answer"):
                if field not in row or row[field] is None:
                    issues.append(f"{label}/{qid}: missing {field}")
            image_hash = row.get("image_sha256")
            if (
                not isinstance(image_hash, str)
                or len(image_hash) != 64
                or any(character not in "0123456789abcdef" for character in image_hash)
            ):
                issues.append(f"{label}/{qid}: image_sha256 must be a lowercase SHA-256 hex digest")
            if not isinstance(row.get("prediction_answer"), str) or not isinstance(
                row.get("reference_answer"), str
            ):
                issues.append(f"{label}/{qid}: answers must be strings")
            if (
                not isinstance(row.get("reference_answer"), str)
                or not row["reference_answer"].strip()
            ):
                issues.append(f"{label}/{qid}: reference_answer must be nonempty")
            elif not normalize_answer(row["reference_answer"]):
                issues.append(f"{label}/{qid}: reference_answer is empty after normalization")
            for field in (
                "image_id", "image_path", "question", "model_id", "model_revision", "compute_dtype"
            ):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    issues.append(f"{label}/{qid}: {field} must be a nonempty string")
            prompt_hash = row.get("prompt_sha256")
            if (
                not isinstance(prompt_hash, str)
                or len(prompt_hash) != 64
                or any(character not in "0123456789abcdef" for character in prompt_hash)
            ):
                issues.append(
                    f"{label}/{qid}: prompt_sha256 must be a lowercase SHA-256 hex digest"
                )
            for field in ("seed", "max_pixels", "min_pixels", "max_new_tokens"):
                value = row.get(field)
                if type(value) is not int or (field != "seed" and value <= 0):
                    constraint = "an integer" if field == "seed" else "an integer > 0"
                    issues.append(f"{label}/{qid}: {field} must be {constraint}")
            if (
                type(row.get("max_pixels")) is int
                and type(row.get("min_pixels")) is int
                and row["min_pixels"] > row["max_pixels"]
            ):
                issues.append(f"{label}/{qid}: min_pixels exceeds max_pixels")
            if (
                not isinstance(row.get("generation_config"), dict)
                or not row["generation_config"]
            ):
                issues.append(f"{label}/{qid}: generation_config must be a nonempty object")
            if type(row.get("load_in_4bit")) is not bool:
                issues.append(f"{label}/{qid}: load_in_4bit must be a boolean")
            if row.get("split") not in ("validation", "test"):
                issues.append(f"{label}/{qid}: expected held-out split")
            protocols.append([row.get(field) for field in PROTOCOL_FIELDS])
        if protocols and any(p != protocols[0] for p in protocols):
            issues.append(f"{label}: mixed inference protocols within one file")
        rows = list(indexed.values())
        for field in ("split", "experiment_id", "adapter"):
            # Runs may differ across files, never between rows of one export.
            if rows and any(
                (field in row, row.get(field)) != (field in rows[0], rows[0].get(field))
                for row in rows[1:]
            ):
                issues.append(f"{label}: mixed {field} within one file")
        return indexed

    a = index_records(left, "left")
    b = index_records(right, "right")
    if checkpoint_comparison and a and b:
        if next(iter(a.values())).get("adapter") == next(iter(b.values())).get("adapter"):
            issues.append("Checkpoint comparison requires two distinct adapter paths")
    if expected_validation_ids is not None:
        for label, indexed in (("left", a), ("right", b)):
            missing = sorted(expected_validation_ids - indexed.keys())
            extra = sorted(indexed.keys() - expected_validation_ids)
            wrong_split = sorted(
                qid for qid, row in indexed.items() if row.get("split") != expected_split
            )
            if missing:
                issues.append(
                    f"{label}: missing {len(missing)} expected {expected_split} question IDs; "
                    f"first: {', '.join(missing[:5])}"
                )
            if extra:
                issues.append(
                    f"{label}: has {len(extra)} question IDs outside the "
                    f"{expected_split} manifest; "
                    f"first: {', '.join(extra[:5])}"
                )
            if wrong_split:
                issues.append(
                    f"{label}: {len(wrong_split)} predictions are not marked {expected_split}; "
                    f"first: {', '.join(wrong_split[:5])}"
                )
    if expected_validation_annotations is not None:
        for label, indexed, experiment_id in (
            ("left", a, expected_experiments[0]),
            ("right", b, expected_experiments[1]),
        ):
            adapter_required = experiment_id != "M0_PRETRAINED"
            adapters = set()
            for qid, row in indexed.items():
                if row.get("experiment_id") != experiment_id:
                    issues.append(f"{label}/{qid}: expected experiment_id={experiment_id}")
                adapter = row.get("adapter")
                if adapter_required:
                    if not isinstance(adapter, str) or not adapter.strip():
                        issues.append(
                            f"{label}/{qid}: {experiment_id.split('_')[0]} requires "
                            "a nonempty adapter path"
                        )
                    else:
                        adapters.add(adapter)
                elif adapter is not None:
                    issues.append(f"{label}/{qid}: M0 must not load an adapter")
            if len(adapters) > 1:
                issues.append(f"{label}: mixed adapter paths within one file")
            for qid, row in indexed.items():
                reference = expected_validation_annotations.get(qid)
                if reference is None:
                    continue
                for prediction_field, annotation_field in (
                    ("image_id", "image_id"),
                    ("image_path", "image_path"),
                    ("question", "question"),
                    ("reference_answer", "answer"),
                    ("reference_explanations", "explanations"),
                    ("split", "split"),
                ):
                    if row.get(prediction_field) != reference.get(annotation_field):
                        issues.append(
                            f"{label}/{qid}: {prediction_field} differs from "
                            f"{expected_split} annotation"
                        )
                # Compare image digests when the annotations include them.
                if "image_sha256" in reference and row.get("image_sha256") != reference[
                    "image_sha256"
                ]:
                    issues.append(
                        f"{label}/{qid}: image_sha256 differs from {expected_split} annotation"
                    )
    if a.keys() != b.keys():
        issues.append("Question ID sets differ; refusing intersection-only scoring")
    for qid in sorted(a.keys() & b.keys()):
        for field in (*REFERENCE_FIELDS, *PROTOCOL_FIELDS):
            if a[qid].get(field) != b[qid].get(field):
                issues.append(f"{qid}: mismatch {field}")
    if issues:
        return {"comparable": False, "issues": issues, "scores": None}

    # Match by question ID because the two files may use different row orders.
    def score(indexed):
        rows = list(indexed.values())
        return {
            "exact_match": mean(
                exact_match(r["prediction_answer"], r["reference_answer"]) for r in rows
            ),
            "token_f1": mean(token_f1(r["prediction_answer"], r["reference_answer"]) for r in rows),
        }

    scores = {"left": score(a), "right": score(b)}
    explanation_issues = []
    for qid in sorted(a):
        if a[qid].get("reference_explanations") != b[qid].get("reference_explanations"):
            explanation_issues.append(f"{qid}: mismatch reference_explanations")
    explanation_values = {}
    for label, indexed in (("left", a), ("right", b)):
        values = []
        for qid, row in indexed.items():
            try:
                values.append(
                    rouge_l_f1(row.get("prediction_explanation"), row.get("reference_explanations"))
                )
            except ValueError as error:
                explanation_issues.append(f"{label}/{qid}: {error}")
        explanation_values[label] = values
    explanation = {
        "comparable": not explanation_issues,
        "issues": explanation_issues,
        "scores": None,
        "delta_right_minus_left": None,
    }
    if not explanation_issues:
        explanation["scores"] = {
            label: mean(values) for label, values in explanation_values.items()
        }
        explanation["delta_right_minus_left"] = (
            explanation["scores"]["right"] - explanation["scores"]["left"]
        )
    return {
        "comparable": True,
        "canonical_references_checked": expected_validation_annotations is not None,
        "run_identity_metadata_checked": expected_validation_annotations is not None,
        "expected_split": expected_split if expected_validation_annotations is not None else None,
        "expected_experiments": (
            list(expected_experiments) if expected_validation_annotations is not None else None
        ),
        **({"comparison_kind": "m1_checkpoint_validation"} if checkpoint_comparison else {}),
        "n": len(a),
        "issues": [],
        "scores": scores,
        "delta_right_minus_left": {
            k: scores["right"][k] - scores["left"][k] for k in scores["left"]
        },
        "explanation_rouge_l_f1": explanation,
        "limitations": (
            "Checks exporter-recorded image SHA-256 equality, not the underlying files. "
            "Without official held-out annotations, run identity and canonical references "
            "are not verified. Exporter metadata is not cryptographic proof of model loading. "
            "Explanation ROUGE-L is lexical overlap, not visual grounding. Not a significance "
            "test; does not verify training leakage or truthful metadata."
        ),
    }
