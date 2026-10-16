"""Reproducible semantic metrics for generated explanations.

BERTScore complements lexical overlap. It does not check whether a statement is
supported by the image, so visual review remains a separate part of the protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable

DEFAULT_BERTSCORE_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_BERTSCORE_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
DEFAULT_BERTSCORE_LAYERS = 9
EXPECTED_BERTSCORE_VERSION = "0.3.13"
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"

_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def prepare_bertscore_inputs(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    """Validate full prediction rows and return IDs, images, candidates, references."""
    if not records:
        raise ValueError("No prediction records to score")

    question_ids: list[str] = []
    image_ids: list[str] = []
    candidates: list[str] = []
    references: list[list[str]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        question_id = record.get("question_id")
        image_id = record.get("image_id")
        candidate = record.get("prediction_explanation")
        row_references = record.get("reference_explanations")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Record {index} question_id must be a nonempty string")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"Record {index} image_id must be a nonempty string")
        if not isinstance(candidate, str):
            raise ValueError(f"Record {index} prediction_explanation must be a string")
        if (
            not isinstance(row_references, list)
            or not row_references
            or any(
                not isinstance(reference, str) or not reference.strip()
                for reference in row_references
            )
        ):
            raise ValueError(
                f"Record {index} reference_explanations must be nonempty strings"
            )
        question_ids.append(question_id)
        image_ids.append(image_id)
        candidates.append(candidate)
        references.append(row_references)

    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question_id in prediction records")
    return question_ids, image_ids, candidates, references


def resolve_bertscore_model(
    model_id: str,
    revision: str,
    *,
    cache_dir: Path | None = None,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Resolve an immutable Hugging Face snapshot for the semantic evaluator."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("BERTScore model ID must be a nonempty string")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("BERTScore model revision must be a 40-character commit SHA")
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Install the semantic evaluator with: pip install -e '.[semantic-eval]'"
            ) from error
        downloader = snapshot_download

    resolved = Path(
        downloader(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=list(_MODEL_FILES),
        )
    )
    if not resolved.is_dir():
        raise ValueError(f"Resolved BERTScore snapshot is not a directory: {resolved}")
    if not (resolved / "config.json").is_file():
        raise ValueError(f"Resolved BERTScore snapshot has no config.json: {resolved}")
    if not (resolved / "model.safetensors").is_file():
        raise ValueError(f"Resolved BERTScore snapshot has no model weights: {resolved}")
    return resolved


def audit_bertscore_lengths(
    records: list[dict[str, Any]], *, model_path: Path, tokenizer: Any = None,
) -> dict[str, int]:
    """Reject texts the BERTScore backend would silently truncate."""
    ids, _, candidates, references = prepare_bertscore_inputs(records)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), use_fast=False, local_files_only=True,
        )
    limit = tokenizer.model_max_length
    if type(limit) is not int or not 0 < limit < 1_000_000:
        raise ValueError("BERTScore tokenizer has no finite model_max_length")
    maxima = {"candidate_max_tokens": 0, "reference_max_tokens": 0}
    reference_count = 0
    for qid, candidate, row_references in zip(ids, candidates, references, strict=True):
        for label, texts in (("candidate", [candidate]), ("reference", row_references)):
            for text in texts:
                # Match sent_encode's whitespace handling, but never truncate.
                length = len(tokenizer.encode(
                    text.strip(), add_special_tokens=True, truncation=False,
                ))
                maxima[f"{label}_max_tokens"] = max(maxima[f"{label}_max_tokens"], length)
                reference_count += int(label == "reference")
                if length > limit:
                    raise ValueError(
                        f"BERTScore would truncate {label} for {qid}: "
                        f"{length} > {limit} tokens; no score was produced"
                    )
    return {
        "n": len(ids), "reference_count": reference_count,
        "model_max_length": limit, "truncated_texts": 0, **maxima,
    }


def _as_finite_vector(values: Any, *, name: str, expected_length: int) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"BERTScore {name} returned the wrong number of values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"BERTScore {name} returned a non-finite value")
    return result


def score_bertscore(
    records: list[dict[str, Any]],
    *,
    model_path: Path,
    model_id: str = DEFAULT_BERTSCORE_MODEL,
    model_revision: str = DEFAULT_BERTSCORE_REVISION,
    num_layers: int = DEFAULT_BERTSCORE_LAYERS,
    batch_size: int = 16,
    device: str = "cpu",
    scorer: Callable[..., Any] | None = None,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Score every row and return aggregate, per-item, and protocol metadata."""
    question_ids, image_ids, candidates, references = prepare_bertscore_inputs(records)
    if not model_path.is_dir():
        raise ValueError(f"BERTScore model path is not a directory: {model_path}")
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("BERTScore num_layers must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("BERTScore batch_size must be a positive integer")
    if device not in ("cpu", "cuda"):
        raise ValueError("BERTScore device must be cpu or cuda")

    try:
        bert_score_version = version_reader("bert-score")
        transformers_version = version_reader("transformers")
        torch_version = version_reader("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install the semantic evaluator with: pip install -e '.[semantic-eval]'"
        ) from error
    if bert_score_version != EXPECTED_BERTSCORE_VERSION:
        raise RuntimeError(
            f"Expected bert-score {EXPECTED_BERTSCORE_VERSION}, got {bert_score_version}"
        )
    if transformers_version != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Expected transformers {EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}"
        )

    if scorer is None:
        try:
            from bert_score import score as scorer
        except ImportError as error:
            raise RuntimeError(
                "Install the semantic evaluator with: pip install -e '.[semantic-eval]'"
            ) from error

    raw = scorer(
        candidates,
        references,
        model_type=str(model_path),
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
        idf=False,
        lang="vi",
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
        return_hash=True,
        verbose=False,
    )
    if (
        not isinstance(raw, tuple)
        or len(raw) != 2
        or not isinstance(raw[0], tuple)
        or len(raw[0]) != 3
    ):
        raise ValueError("BERTScore backend must return ((precision, recall, F1), hash)")
    precision = _as_finite_vector(raw[0][0], name="precision", expected_length=len(records))
    recall = _as_finite_vector(raw[0][1], name="recall", expected_length=len(records))
    f1 = _as_finite_vector(raw[0][2], name="F1", expected_length=len(records))
    backend_hash = raw[1]
    if not isinstance(backend_hash, str) or not backend_hash:
        raise ValueError("BERTScore backend returned an invalid settings hash")
    version_match = re.search(
        r"_version=([^()]+)\(hug_trans=([^)]+)\)$",
        backend_hash,
    )
    settings_marker = f"_L{num_layers}"
    marker_index = backend_hash.rfind(settings_marker)
    if version_match is None or marker_index < 0:
        raise ValueError("BERTScore backend returned an unrecognized settings hash")
    backend_bert_score_version, backend_transformers_version = version_match.groups()
    if backend_transformers_version != transformers_version:
        raise RuntimeError(
            "BERTScore backend and installed transformers versions do not match: "
            f"{backend_transformers_version} != {transformers_version}"
        )
    # The 0.3.13 wheel reports 0.3.12 in its internal settings string. Preserve
    # both values and replace the machine-specific cache path with the model ID.
    backend_settings = model_id + backend_hash[marker_index:]

    protocol = {
        "metric": "bertscore",
        "model_id": model_id,
        "model_revision": model_revision,
        "num_layers": num_layers,
        "language": "vi",
        "bert_score_version": bert_score_version,
        "backend_reported_bert_score_version": backend_bert_score_version,
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "python_version": platform.python_version(),
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "reference_aggregation": "best_match_across_all_references",
        "device": device,
        "batch_size": batch_size,
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    items = [
        {
            "question_id": question_id,
            "image_id": image_id,
            "precision": item_precision,
            "recall": item_recall,
            "f1": item_f1,
        }
        for question_id, image_id, item_precision, item_recall, item_f1 in zip(
            question_ids, image_ids, precision, recall, f1, strict=True
        )
    ]
    return {
        "status": "scored",
        "n": len(items),
        "mean_precision": mean(precision),
        "mean_recall": mean(recall),
        "mean_f1": mean(f1),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "backend_settings": backend_settings,
        "items": items,
        "limitations": (
            "BERTScore measures text-semantic similarity to references; it does not verify "
            "visual grounding or factual support from the image."
        ),
    }


def _validated_semantic_report(
    report: dict[str, Any],
    *,
    label: str,
    expected_predictions_sha256: str,
    expected_split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not isinstance(report, dict):
        raise ValueError(f"{label} semantic report must be an object")
    if report.get("predictions_sha256") != expected_predictions_sha256:
        raise ValueError(f"{label} semantic report is bound to different predictions")
    if report.get("split") != expected_split:
        raise ValueError(f"{label} semantic report has the wrong split")
    semantic = report.get("explanation_bertscore")
    if not isinstance(semantic, dict) or semantic.get("status") != "scored":
        raise ValueError(f"{label} semantic report is not complete")
    protocol = semantic.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError(f"{label} semantic report has no protocol")
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if semantic.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"{label} semantic protocol hash does not match its payload")
    items = semantic.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{label} semantic report has no per-item scores")
    if semantic.get("n") != len(items) or report.get("n") != len(items):
        raise ValueError(f"{label} semantic report count mismatch")

    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label} semantic item {index} must be an object")
        question_id = item.get("question_id")
        image_id = item.get("image_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{label} semantic item {index} has an invalid question_id")
        if question_id in indexed:
            raise ValueError(f"{label} semantic report has duplicate question_id {question_id}")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{label}/{question_id} has an invalid image_id")
        for field in ("precision", "recall", "f1"):
            value = item.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{label}/{question_id} has an invalid {field}")
            if not -1.000001 <= value <= 1.000001:
                raise ValueError(f"{label}/{question_id} {field} is outside cosine bounds")
        indexed[question_id] = item

    for field, item_field in (
        ("mean_precision", "precision"),
        ("mean_recall", "recall"),
        ("mean_f1", "f1"),
    ):
        stored = semantic.get(field)
        recomputed = mean(item[item_field] for item in items)
        if type(stored) not in (int, float) or not math.isclose(
            stored, recomputed, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} semantic aggregate {field} does not match its items")
    return indexed, protocol


def compare_bertscore_reports(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_predictions_sha256: str,
    right_predictions_sha256: str,
    expected_split: str,
    expected_question_images: dict[str, str],
) -> tuple[dict[str, Any], list[float], list[str]]:
    """Validate and compare two item-level BERTScore reports by question ID."""
    left_items, left_protocol = _validated_semantic_report(
        left,
        label="left",
        expected_predictions_sha256=left_predictions_sha256,
        expected_split=expected_split,
    )
    right_items, right_protocol = _validated_semantic_report(
        right,
        label="right",
        expected_predictions_sha256=right_predictions_sha256,
        expected_split=expected_split,
    )
    if left_protocol != right_protocol:
        raise ValueError("Semantic reports use different BERTScore protocols")
    if left_items.keys() != right_items.keys():
        raise ValueError("Semantic reports cover different question IDs")
    if left_items.keys() != expected_question_images.keys():
        raise ValueError("Semantic reports do not cover the paired prediction IDs")

    question_ids = sorted(left_items)
    image_ids: list[str] = []
    differences: list[float] = []
    for question_id in question_ids:
        left_item = left_items[question_id]
        right_item = right_items[question_id]
        if left_item["image_id"] != right_item["image_id"]:
            raise ValueError(f"Semantic reports disagree on image_id for {question_id}")
        if left_item["image_id"] != expected_question_images[question_id]:
            raise ValueError(f"Semantic report image_id differs from predictions for {question_id}")
        image_ids.append(left_item["image_id"])
        differences.append(right_item["f1"] - left_item["f1"])

    scores = {
        "left": {
            field: mean(item[field] for item in left_items.values())
            for field in ("precision", "recall", "f1")
        },
        "right": {
            field: mean(item[field] for item in right_items.values())
            for field in ("precision", "recall", "f1")
        },
    }
    comparison = {
        "comparable": True,
        "n": len(question_ids),
        "protocol": left_protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(left_protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "scores": scores,
        "delta_right_minus_left": {
            field: scores["right"][field] - scores["left"][field]
            for field in ("precision", "recall", "f1")
        },
        "limitations": (
            "Paired BERTScore measures text-semantic similarity, not visual grounding or "
            "training-seed stability."
        ),
    }
    return comparison, differences, image_ids


def evaluate_bertscore(
    records: list[dict[str, Any]],
    *,
    model_id: str = DEFAULT_BERTSCORE_MODEL,
    model_revision: str = DEFAULT_BERTSCORE_REVISION,
    num_layers: int = DEFAULT_BERTSCORE_LAYERS,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve the pinned model and score records with an explicit runtime device."""
    prepare_bertscore_inputs(records)
    if device == "auto":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Install the semantic evaluator with: pip install -e '.[semantic-eval]'"
            ) from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_bertscore_model(
        model_id,
        model_revision,
        cache_dir=cache_dir,
    )
    length_audit = audit_bertscore_lengths(records, model_path=model_path)
    result = score_bertscore(
        records,
        model_path=model_path,
        model_id=model_id,
        model_revision=model_revision,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
    )
    result["length_audit"] = length_audit
    return result
