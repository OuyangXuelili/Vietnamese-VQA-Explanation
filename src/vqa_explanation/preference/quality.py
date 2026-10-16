"""Screen preference shortcuts and check image-based review records."""

import hashlib
import json
import random
import re
from collections import Counter

import numpy as np

from vqa_explanation.evaluation.answer_metrics import normalize_answer
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.preference.schema import PreferencePair


def candidate_origin(inputs):
    """Recover adapter identity from the candidate export's saved weight hashes."""
    files = inputs.get("adapter_files")
    if (inputs.get("experiment_id") != "M1_SFT" or inputs.get("split") != "train"
            or type(inputs.get("seed")) is not int or inputs["seed"] < 0
            or not isinstance(inputs.get("model_id"), str) or not inputs["model_id"].strip()
            or not isinstance(inputs.get("model_revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", inputs["model_revision"])
            or not isinstance(files, dict)
            or set(files) != {"adapter_config.json", "adapter_model.safetensors"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                   for v in files.values())):
        raise ValueError("Candidate manifest needs a pinned train M1 adapter")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return {"model_id": inputs["model_id"], "model_revision": inputs["model_revision"],
            "checkpoint_sha256": digest, "seed": inputs["seed"]}


def pair_digest(pair):
    fields = {key: pair[key] for key in (
        "pair_id", "source_question_id", "image_id", "image_sha256", "chosen", "rejected"
    )}
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_pairs(pairs):
    if not pairs:
        raise ValueError("No preference pairs")
    seen = set()
    for row in pairs:
        pair = PreferencePair.from_dict(row)
        if pair.pair_id in seen:
            raise ValueError("Duplicate pair_id")
        seen.add(pair.pair_id)
        if row.get("split") != "train" or row.get("review_status") != "accepted":
            raise ValueError("Quality checks require accepted train pairs")
        for field in ("image_id", "image_sha256"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}")
        outputs = [parse_prediction(row[side]) for side in ("chosen", "rejected")]
        if not all(output.format_valid for output in outputs) or (
            normalize_answer(outputs[0].answer) != normalize_answer(outputs[1].answer)
        ):
            raise ValueError("Both completions need the same answer and valid format")


def review_template(pairs):
    validate_pairs(pairs)
    return [{
        "pair_id": row["pair_id"], "pair_sha256": pair_digest(row), "review_version": 2,
        "status": "pending",
        "reviewer_type": None, "visible_evidence": "", "rejected_span": "",
        "answer_correct": None, "chosen_supported": None, "rejected_unsupported": None,
        "both_linguistically_plausible": None, "language_only_preference": None,
        "contrast_relevant": None, "relevance_reason": "", "text_only_reason": "",
    } for row in pairs]


def assess_reviews(pairs, reviews, minimum_pairs=500, *, required_version=1):
    """Check supplied records; these checks cannot judge an image or a rationale."""
    validate_pairs(pairs)
    if type(minimum_pairs) is not int or minimum_pairs < 1:
        raise ValueError("minimum_pairs must be a positive integer")
    if type(required_version) is not int or required_version not in {1, 2}:
        raise ValueError("Unsupported required review version")
    known = {row["pair_id"]: row for row in pairs}
    seen, issues, details = set(), [], []
    confirmed = 0
    for review in reviews:
        pid = review.get("pair_id")
        if pid not in known or pid in seen:
            raise ValueError("Unknown or duplicate quality review pair_id")
        seen.add(pid)
        pair = known[pid]
        if review.get("pair_sha256") != pair_digest(pair):
            raise ValueError(f"Quality review is stale: {pid}")
        checks = {
            "status": review.get("status") == "confirmed",
            "reviewer_type": review.get("reviewer_type") in {"human", "ai"},
            "visual_assessment": all(review.get(key) is True for key in (
                "answer_correct", "chosen_supported", "rejected_unsupported",
                "both_linguistically_plausible",
            )),
            "language_only_preference": review.get("language_only_preference") == "undetermined",
            "visible_evidence": isinstance(review.get("visible_evidence"), str)
            and bool(review["visible_evidence"].strip()),
            "rejected_span": isinstance(review.get("rejected_span"), str)
            and bool(review["rejected_span"].strip())
            and review["rejected_span"] in parse_prediction(pair["rejected"]).explanation,
        }
        if required_version == 2 or review.get("review_version") == 2:
            checks.update({
                "review_version": type(review.get("review_version")) is int
                and review["review_version"] == 2,
                "contrast_relevant": review.get("contrast_relevant") is True,
                **{key: isinstance(review.get(key), str) and bool(review[key].strip())
                   for key in ("relevance_reason", "text_only_reason")},
            })
        if all(checks.values()):
            confirmed += 1
        else:
            issues.append(pid)
            details.append({"pair_id": pid,
                            "failed_checks": [name for name, passed in checks.items()
                                              if not passed]})
    missing = sorted(set(known) - seen)
    unique_images = len({r["image_id"] for r in pairs})
    unique_paths = len({r["image_path"] for r in pairs})
    ready = (len(pairs) >= minimum_pairs and confirmed == len(pairs)
             and unique_images == unique_paths == len(pairs))
    return {
        "ready": ready, "pairs": len(pairs), "minimum_pairs": minimum_pairs,
        "confirmed": confirmed, "missing": missing, "needs_review": issues,
        "issue_details": details, "required_review_version": required_version,
        "unique_images": unique_images,
        "verification_scope": "review_record_consistency",
        "limitations": "Field checks cannot verify visual truth or reviewer independence.",
    }


def _features(text, dimensions):
    tokens = re.findall(r"\w+", text.casefold())
    terms = tokens + [a + " " + b for a, b in zip(tokens, tokens[1:], strict=False)]
    values = np.zeros(dimensions, dtype=np.float64)
    for term, count in Counter(terms).items():
        digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % (dimensions - 1)
        values[index] += 1 + np.log(count)
    norm = np.linalg.norm(values[:-1])
    if norm:
        values[:-1] /= norm
    values[-1] = np.log1p(len(tokens)) / 5
    return values


def text_only_screen(pairs, seed=42, folds=5, bootstrap=2000):
    """Held-out linear ranking from completions only; no visual-quality claim."""
    validate_pairs(pairs)
    if type(folds) is not int or folds < 2 or type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Require at least two folds and a positive bootstrap count")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    # Group path aliases and repeated image IDs into the same held-out fold.
    groups = []
    for row in sorted(pairs, key=lambda r: r["pair_id"]):
        matches = [g for g in groups if any(
            row["image_id"] == p["image_id"] or row["image_path"] == p["image_path"]
            for p in g
        )]
        if not matches:
            groups.append([row])
        else:
            merged = [row, *[p for g in matches for p in g]]
            groups = [g for g in groups if all(g is not m for m in matches)] + [merged]
    if len(groups) < folds:
        raise ValueError("Too few independent image groups for the requested folds")
    groups.sort(key=lambda g: min(p["pair_id"] for p in g))
    random.Random(seed).shuffle(groups)
    vectors, image_groups = [], []
    for index, group in enumerate(groups):
        for row in group:
            difference = (_features(row["chosen"], 512) - _features(row["rejected"], 512))
            vectors.append((row, difference, index % folds))
            image_groups.append(index)
    matrix = np.array([v[1] for v in vectors])
    scores = np.zeros(len(vectors))
    for fold in range(folds):
        heldout = np.array([v[2] == fold for v in vectors])
        train = matrix[~heldout]
        weights = np.zeros(matrix.shape[1])
        # Pairwise logistic ranking; fixed settings, no test-fold tuning.
        for _ in range(200):
            margins = np.clip(train @ weights, -30, 30)
            gradient = -(train.T @ (1 / (1 + np.exp(margins)))) / len(train)
            weights -= 0.5 * (gradient + 0.05 * weights)
        scores[heldout] = matrix[heldout] @ weights
    wins = np.where(scores > 1e-10, 1.0, np.where(scores < -1e-10, 0.0, 0.5))
    grouped = [wins[np.array(image_groups) == index] for index in range(len(groups))]
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap):
        drawn = [grouped[rng.randrange(len(groups))] for _ in groups]
        samples.append(float(np.concatenate(drawn).mean()))
    return {
        "method": "hashed_unigram_bigram_pairwise_logistic_group_cv_v1",
        "folds": folds, "seed": seed, "pairs": len(pairs), "image_groups": len(groups),
        "pair_accuracy": float(wins.mean()), "chance": 0.5,
        "image_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_resamples": bootstrap,
        "error_types": dict(Counter(r["error_type"] for r in pairs)),
        "rows": [{"pair_id": v[0]["pair_id"], "fold": v[2],
                  "group": image_groups[i], "margin": float(scores[i]),
                  "pair_credit": float(wins[i])} for i, v in enumerate(vectors)],
        "limitations": "A text-only screening baseline, not Qwen and not evidence that "
                       "any trained model ignores images. A low score does not certify hard pairs.",
    }
