import copy

import pytest

from vqa_explanation.evaluation.image_check_report import IMAGE_ARMS, paired_image_results


def fixture_predictions():
    original = [{"question_id": str(i), "image_id": str(i), "split": "validation",
                 "answer": "đỏ"} for i in range(4)]
    def row(i, answer):
        return {"question_id": str(i), "prediction_answer": answer,
                "prediction_explanation": "Có màu đỏ."}
    predictions = {a: {"correct": [row(i, "đỏ") for i in range(4)],
                        "permuted": [row(i, "xanh") for i in range(4)]} for a in IMAGE_ARMS}
    return original, predictions


def test_between_arm_bootstrap_pairs_by_id_and_preserves_inputs():
    original, predictions = fixture_predictions()
    predictions["m3"]["permuted"][0]["prediction_answer"] = "đỏ"
    before = copy.deepcopy(predictions)
    report, items = paired_image_results(predictions, original, resamples=100)
    assert predictions == before
    assert len(items) == 12
    assert report["scores"]["m2"]["em_drop"] == 1
    assert report["scores"]["m3"]["em_drop"] == 0.75
    delta = report["comparisons"]["m3_minus_m2"]["metrics"]["em_drop"]
    assert delta["delta_right_minus_left"] == -0.25
    for arm in IMAGE_ARMS:
        for rows in predictions[arm].values():
            rows.reverse()
    assert paired_image_results(predictions, original, resamples=100)[0] == report


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_id", "wrong_split",
                                   "duplicate_image", "missing_arm", "missing_condition"])
def test_invalid_pairs_are_rejected(change):
    original, predictions = fixture_predictions()
    if change == "duplicate":
        predictions["m2"]["correct"][1] = predictions["m2"]["correct"][0]
    elif change == "missing":
        predictions["m3"]["permuted"].pop()
    elif change == "wrong_id":
        predictions["m1"]["correct"][0]["question_id"] = "unknown"
    elif change == "wrong_split":
        original[0]["split"] = "test"
    elif change == "duplicate_image":
        original[0]["image_id"] = original[1]["image_id"]
    elif change == "missing_arm":
        predictions.pop("m1")
    else:
        predictions["m1"].pop("correct")
    with pytest.raises(ValueError):
        paired_image_results(predictions, original, resamples=100)


def test_explanation_change_is_not_a_correctness_score():
    original, predictions = fixture_predictions()
    predictions["m3"]["permuted"][0]["prediction_explanation"] = "  CÓ MÀU ĐỎ. "
    predictions["m3"]["permuted"][1]["prediction_explanation"] = "Có màu xanh."
    report, _ = paired_image_results(predictions, original, resamples=100)
    assert report["scores"]["m3"]["explanation_change_rate"] == 0.25
    assert "not visual correctness" in " ".join(report["limitations"])
