from copy import deepcopy

import pytest

from vqa_explanation.evaluation.validation_report import analyze_predictions, render_cases


def data():
    left, right, refs = [], [], []
    for i, (a, b) in enumerate(((1, 1), (0, 1), (1, 0), (0, 0))):
        ref = {"question_id": str(i), "image_id": str(i), "image_path": f"images/{i}.jpg",
               "question": "Con gì?", "answer": "mèo", "explanations": ["Có mèo."],
               "split": "validation"}
        row = {**ref, "reference_answer": ref["answer"],
               "reference_explanations": ref["explanations"], "image_sha256": "a" * 64,
               "model_id": "model", "model_revision": "revision", "seed": 42,
               "max_pixels": 200704, "min_pixels": 200704, "max_new_tokens": 128,
               "prompt_sha256": "b" * 64, "generation_config": {"do_sample": False},
               "load_in_4bit": True, "compute_dtype": "float16", "format_valid": True,
               "prediction_explanation": "Có mèo."}
        left.append({**row, "prediction_answer": "mèo" if a else "chó",
                     "experiment_id": "M0_PRETRAINED", "adapter": None})
        right.append({**row, "prediction_answer": "mèo" if b else "chó",
                      "experiment_id": "M1_SFT", "adapter": "checkpoint-3683"})
        refs.append(ref)
    return left, right, refs


def test_paired_report_keeps_regressions_and_deterministic_case_selection():
    left, right, refs = data()
    report = analyze_predictions(left, right, refs, resamples=50)
    assert report["em_transitions"] == dict(both_match=1, gain=1, loss=1, neither_match=1)
    assert report["scores"]["M0"]["exact_match"] == .5
    assert report["scores"]["M1"]["exact_match"] == .5
    assert report["bootstrap"]["metrics"]["exact_match"]["delta_right_minus_left"] == 0
    reordered = analyze_predictions(left[::-1], right[::-1], refs[::-1], resamples=50)
    assert report["case_selection"] == reordered["case_selection"]
    assert report["bootstrap"] == reordered["bootstrap"]


@pytest.mark.parametrize("fault", ["missing", "protocol", "test", "duplicate", "format"])
def test_report_refuses_incomplete_or_mismatched_sources(fault):
    left, right, refs = data()
    if fault == "missing":
        right.pop()
    elif fault == "protocol":
        right[0]["max_new_tokens"] = 256
    elif fault == "test":
        refs[0]["split"] = "test"
    elif fault == "duplicate":
        refs.append(deepcopy(refs[0]))
    else:
        right[0].pop("format_valid")
    with pytest.raises(ValueError):
        analyze_predictions(left, right, refs, resamples=50)


def test_case_html_escapes_generated_text():
    left, right, refs = data()
    right[0]["prediction_explanation"] = "<script>alert(1)</script>"
    report = analyze_predictions(left, right, refs, resamples=50)
    html = render_cases(report, {r["question_id"]: r for r in left},
                        {r["question_id"]: r for r in right},
                        {r["question_id"]: r for r in refs})
    assert "<script>" not in html and "&lt;script&gt;" in html
