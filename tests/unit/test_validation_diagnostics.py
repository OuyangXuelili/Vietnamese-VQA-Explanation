import json
from copy import deepcopy

import pytest

from tests.unit.test_study_report import semantic_data, study_data
from vqa_explanation.evaluation.validation_diagnostics import (
    SEEDS,
    analyze_validation,
    question_type,
    transition,
)
from vqa_explanation.evaluation.validation_diagnostics_bundle import render_markdown, transition_svg


def inputs():
    predictions, semantic, hashes = {}, {}, {}
    for seed in SEEDS:
        predictions[seed], annotations = study_data()
        for rows in predictions[seed].values():
            for row in rows:
                row["raw_output"] = (f"Đáp án: {row['prediction_answer']}\n"
                                     f"Giải thích: {row['prediction_explanation']}")
        semantic[seed], hashes[seed] = semantic_data(predictions[seed])
    return predictions, annotations, semantic, hashes


def report():
    return analyze_validation(*inputs(), resamples=10)


@pytest.mark.parametrize(("question", "expected"), [
    ("Có bao nhiêu con chó?", "count"), ("Chiếc áo màu gì?", "color"),
    ("Ảnh được chụp ở đâu?", "location"), ("Người này đang làm gì?", "action"),
    ("Đây có phải là ô tô không?", "yes_no"), ("Đây là phòng gì?", "object"),
    ("Nêu chi tiết trong bức hình.", "other"),
])
def test_question_groups_are_prediction_independent(question, expected):
    assert question_type(question) == expected


def test_counts_baselines_seed_sd_and_transitions_are_not_pseudoreplicated():
    result = report()
    assert result["n"] == result["n_images"] == 4
    assert result["unique_inference_runs"] == 8
    assert result["aggregate"]["m0"]["exact_match"]["sd"] is None
    assert result["aggregate"]["m2"]["exact_match"]["sd"] == 0
    assert result["by_seed"]["42"]["em_transitions_m2_m3"] == {
        "both_match": 1, "gain": 1, "loss": 1, "neither_match": 1}
    assert sum(g["n"] for g in result["groups"] if g["dimension"] == "question_type") == 4
    assert sum(g["n"] for g in result["groups"] if g["dimension"] == "reference_length") == 4
    assert result["answer_stability"]["m3"]["same_normalized_answer_n"] == 4
    assert result["answer_stability"]["m3"]["em_patterns"] == {
        "all_match": 2, "mixed": 0, "none_match": 2}


def test_input_order_does_not_change_report_or_case_selection():
    predictions, annotations, semantic, hashes = inputs()
    first = analyze_validation(predictions, annotations, semantic, hashes, resamples=10)
    second = analyze_validation({s: {a: rows[::-1] for a, rows in p.items()}
                                 for s, p in predictions.items()}, annotations[::-1],
                                semantic, hashes, resamples=10)
    assert first == second


@pytest.mark.parametrize("fault", ["test", "seed", "duplicate", "missing", "protocol",
                                    "baseline", "image", "semantic"])
def test_rejects_mixed_or_incomplete_evidence(fault):
    predictions, annotations, semantic, hashes = inputs()
    if fault == "test":
        annotations[0]["split"] = "test"
    elif fault == "seed":
        predictions.pop(44)
    elif fault == "duplicate":
        annotations.append(deepcopy(annotations[0]))
    elif fault == "missing":
        predictions[43]["m3"].pop()
    elif fault == "protocol":
        predictions[43]["m3"][0]["max_pixels"] = 100
    elif fault == "baseline":
        hashes[43]["m0"] = "0" * 64
        semantic[43]["m0"]["predictions_sha256"] = hashes[43]["m0"]
    elif fault == "image":
        predictions[43]["m3"][0]["image_sha256"] = "0" * 64
    else:
        semantic[44]["m3"]["explanation_bertscore"]["items"].pop()
    with pytest.raises(ValueError):
        analyze_validation(predictions, annotations, semantic, hashes, resamples=10)


def test_report_does_not_mutate_or_rewrite_raw_outputs():
    args = inputs()
    args[0][42]["m3"][0]["raw_output"] = '<script>alert("unsafe")</script>'
    before = deepcopy(args)
    result = analyze_validation(*args, resamples=10)
    assert args == before
    assert result["cases"][0]["seeds"]["42"]["m3"]["raw_output"].startswith('<script>')
    assert "chấm mù" in render_markdown(result)
    assert "validation" in transition_svg(result)
    assert "test" in json.dumps(result["limitations"])


def test_em_transition_directions():
    assert transition(0, 1) == "gain"
    assert transition(1, 0) == "loss"
    assert transition(1, 1) == "both_match"
    assert transition(0, 0) == "neither_match"
