from vqa_explanation.evaluation.answer_metrics import exact_match, token_f1


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("Con mèo!", "con mèo") == 1.0


def test_token_f1_handles_partial_overlap() -> None:
    score = token_f1("một con mèo đen", "con mèo")
    assert 0.0 < score < 1.0

