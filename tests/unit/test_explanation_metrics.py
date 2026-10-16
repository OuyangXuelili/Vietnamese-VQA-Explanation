import pytest

from vqa_explanation.evaluation.explanation_metrics import rouge_l_f1


def test_rouge_l_normalizes_vietnamese_nfc_but_keeps_accents():
    assert rouge_l_f1("mèo đen", ["me\u0300o đen"]) == 1.0
    assert rouge_l_f1("meo den", ["mèo đen"]) == 0.0


def test_rouge_l_uses_order_and_best_reference():
    assert rouge_l_f1("mèo trên bàn", ["bàn trên mèo"]) == pytest.approx(1 / 3)
    assert rouge_l_f1("mèo trên bàn", ["chó dưới bàn", "mèo trên bàn!"]) == 1.0
    assert rouge_l_f1("mèo mèo", ["mèo"]) == pytest.approx(2 / 3)


def test_empty_prediction_scores_zero_but_invalid_references_fail():
    assert rouge_l_f1("", ["Có mèo."]) == 0.0
    for references in ([], [""], ["!!!"], ["Có mèo", None]):
        with pytest.raises(ValueError, match="Reference"):
            rouge_l_f1("", references)
    with pytest.raises(ValueError, match="Prediction"):
        rouge_l_f1(None, ["Có mèo"])
