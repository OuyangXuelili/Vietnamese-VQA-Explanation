"""Validation-only error diagnostics across the three declared training seeds."""

import re
import unicodedata
from collections import Counter
from statistics import mean, stdev

from vqa_explanation.evaluation.answer_metrics import exact_match, normalize_answer, token_f1
from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1
from vqa_explanation.evaluation.study_report import analyze_study
from vqa_explanation.inference.study_validation import ARMS

SEEDS = (42, 43, 44)
METRICS = ("exact_match", "token_f1", "rouge_l", "bertscore_f1")
TYPE_LABELS = {
    "count": "Số lượng",
    "color": "Màu sắc",
    "location": "Vị trí/bối cảnh",
    "action": "Hành động",
    "yes_no": "Có/không",
    "object": "Đối tượng/loại",
    "other": "Khác",
}
TRANSITIONS = {
    "both_match": "Cùng khớp EM",
    "gain": "M3 khớp, đối chứng không khớp",
    "loss": "M3 không khớp, đối chứng khớp",
    "neither_match": "Cùng không khớp EM",
}


def question_type(question):
    text = " ".join(unicodedata.normalize("NFC", question).lower().split())
    rules = (
        ("count", r"bao nhiêu|\bmấy\b"),
        ("color", r"\bmàu\b"),
        ("location", r"ở đâu|nơi nào|chỗ nào|vị trí nào"),
        ("action", r"làm gì|đang làm|hoạt động|chơi môn|đang chơi"),
        ("yes_no", r"(?:không|phải chứ|đúng chứ|phải vậy không)[?.!]*$"),
        ("object", r"con gì|cái gì|vật gì|loại nào|loại gì|là gì|phòng gì|môn gì"),
    )
    return next((name for name, pattern in rules if re.search(pattern, text)), "other")


def transition(left, right):
    return (
        "both_match" if left and right else "loss" if left else "gain" if right else "neither_match"
    )


def _mean_sd(values):
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}


def _output(row, semantic_f1):
    return {
        "answer": row["prediction_answer"],
        "explanation": row["prediction_explanation"],
        "raw_output": row["raw_output"],
        "format_valid": row["format_valid"],
        "latency_seconds": row["latency_seconds"],
        "scores": {
            "exact_match": exact_match(row["prediction_answer"], row["reference_answer"]),
            "token_f1": token_f1(row["prediction_answer"], row["reference_answer"]),
            "rouge_l": rouge_l_f1(row["prediction_explanation"], row["reference_explanations"]),
            "bertscore_f1": semantic_f1,
        },
        "explanation_whitespace_tokens": len(row["prediction_explanation"].split()),
    }


def analyze_validation(predictions, annotations, semantic, hashes, *, resamples=2000):
    if set(predictions) != set(SEEDS) or set(semantic) != set(SEEDS) or set(hashes) != set(SEEDS):
        raise ValueError("Require exactly training seeds 42, 43 and 44")
    canonical = {r["question_id"]: r for r in annotations}
    if (
        not canonical
        or len(canonical) != len(annotations)
        or any(r["split"] != "validation" for r in annotations)
    ):
        raise ValueError("Require unique validation annotations, never test")
    indexed, items, global_reports = {}, {}, {}
    for seed in SEEDS:
        global_reports[seed] = analyze_study(
            predictions[seed],
            annotations,
            semantic=semantic[seed],
            prediction_hashes=hashes[seed],
            training_seed=seed,
            resamples=resamples,
        )
        indexed[seed] = {
            a: {r["question_id"]: r for r in rows} for a, rows in predictions[seed].items()
        }
        items[seed] = {
            a: {
                r["question_id"]: r["f1"]
                for r in semantic[seed][a]["explanation_bertscore"]["items"]
            }
            for a in ARMS
        }
        if global_reports[seed]["inference_protocol"] != global_reports[42]["inference_protocol"]:
            raise ValueError("Inference protocols differ across training seeds")
        if global_reports[seed]["semantic_protocol"] != global_reports[42]["semantic_protocol"]:
            raise ValueError("Semantic protocols differ across training seeds")
        for arm in ("m0", "m1"):
            if hashes[seed][arm] != hashes[42][arm]:
                raise ValueError("M0/M1 must be the same fixed baseline exports across seeds")
    cases = []
    for qid in sorted(canonical):
        row = canonical[qid]
        ref_length = mean(len(text.split()) for text in row["explanations"])
        case = {
            "question_id": qid,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "image_sha256": indexed[42]["m0"][qid]["image_sha256"],
            "question": row["question"],
            "reference_answer": row["answer"],
            "reference_explanations": row["explanations"],
            "question_type": question_type(row["question"]),
            "reference_length": "short"
            if ref_length <= 8
            else "medium"
            if ref_length <= 16
            else "long",
            "baselines": {a: _output(indexed[42][a][qid], items[42][a][qid]) for a in ("m0", "m1")},
            "seeds": {},
        }
        for seed in SEEDS:
            variants = {
                a: _output(indexed[seed][a][qid], items[seed][a][qid]) for a in ("m2", "m3")
            }
            for arm in ARMS:
                if indexed[seed][arm][qid]["image_sha256"] != case["image_sha256"]:
                    raise ValueError("Image payload changed between runs")
            m3_em = variants["m3"]["scores"]["exact_match"]
            variants["vs_m2"] = transition(variants["m2"]["scores"]["exact_match"], m3_em)
            variants["vs_m1"] = transition(case["baselines"]["m1"]["scores"]["exact_match"], m3_em)
            case["seeds"][str(seed)] = variants
        cases.append(case)
    grouped = []
    for dimension in ("question_type", "reference_length"):
        for group in sorted({c[dimension] for c in cases}):
            selected = [c for c in cases if c[dimension] == group]
            scores = {
                str(s): {
                    a: {
                        m: mean(
                            (c["baselines"][a] if a in {"m0", "m1"} else c["seeds"][str(s)][a])[
                                "scores"
                            ][m]
                            for c in selected
                        )
                        for m in METRICS
                    }
                    for a in ARMS
                }
                for s in SEEDS
            }
            grouped.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": len(selected),
                    "n_images": len({c["image_id"] for c in selected}),
                    "small_group": len(selected) < 30,
                    "scores": scores,
                    "m3_minus_m2": {
                        m: _mean_sd(
                            [scores[str(s)]["m3"][m] - scores[str(s)]["m2"][m] for s in SEEDS]
                        )
                        for m in METRICS
                    },
                }
            )
    aggregate = {
        a: {
            m: _mean_sd(
                [global_reports[s]["scores"][a][m] for s in (SEEDS if a in {"m2", "m3"} else (42,))]
            )
            for m in METRICS
        }
        for a in ARMS
    }
    stability = {}
    for arm in ("m2", "m3"):
        patterns = Counter()
        identical = 0
        for c in cases:
            outputs = [c["seeds"][str(s)][arm] for s in SEEDS]
            matches = sum(bool(o["scores"]["exact_match"]) for o in outputs)
            patterns[
                "all_match" if matches == 3 else "none_match" if matches == 0 else "mixed"
            ] += 1
            identical += len({normalize_answer(o["answer"]) for o in outputs}) == 1
        stability[arm] = {
            "same_normalized_answer_n": identical,
            "em_patterns": {k: patterns[k] for k in ("all_match", "mixed", "none_match")},
        }
    return {
        "schema": "validation_diagnostics_v1",
        "split": "validation",
        "n": len(cases),
        "n_images": len({c["image_id"] for c in cases}),
        "training_seeds": list(SEEDS),
        "unique_inference_runs": 8,
        "baselines_shared": True,
        "type_labels": TYPE_LABELS,
        "transition_labels": TRANSITIONS,
        "group_method": ("First matching question-text regex; "
                         "not human labels or a benchmark taxonomy"),
        "length_method": "Mean reference explanation whitespace tokens; <=8, 9-16, >16",
        "aggregate": aggregate,
        "groups": grouped,
        "answer_stability": stability,
        "by_seed": {
            str(s): {
                k: global_reports[s][k]
                for k in ("scores", "bootstrap", "diagnostics", "em_transitions_m2_m3")
            }
            for s in SEEDS
        },
        "transitions_vs_m1": {
            str(s): dict(Counter(c["seeds"][str(s)]["vs_m1"] for c in cases)) for s in SEEDS
        },
        "limitations": [
            "Validation exploration only; no test predictions read or generated",
            "EM is reference-string agreement, not visual correctness",
            "ROUGE-L/BERTScore do not establish image-grounded explanations",
            "Question groups are heuristic and descriptive, not confirmatory tests",
            "Seed SD is not a confidence interval; M0/M1 were not retrained three times",
        ],
        "cases": cases,
    }
