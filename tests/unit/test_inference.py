from vqa_explanation.inference.prompting import build_user_prompt, parse_prediction


def test_build_user_prompt_requires_a_question() -> None:
    try:
        build_user_prompt("   ")
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("empty question should fail")


def test_parse_prediction_accepts_expected_format() -> None:
    result = parse_prediction(
        "Đáp án: hai người\nGiải thích: Trong ảnh có hai người đang đứng cạnh xe."
    )
    assert result.answer == "hai người"
    assert result.explanation == "Trong ảnh có hai người đang đứng cạnh xe."
    assert result.format_valid


def test_parse_prediction_marks_unstructured_text_invalid() -> None:
    result = parse_prediction("Tôi nhìn thấy hai người đứng cạnh xe.")
    assert result.answer == ""
    assert result.explanation == ""
    assert not result.format_valid


def test_parse_prediction_does_not_treat_next_label_as_an_empty_answer() -> None:
    result = parse_prediction("Đáp án: \nGiải thích: Có một con mèo.")
    assert result.answer == ""
    assert result.explanation == "Có một con mèo."
    assert not result.format_valid


def test_parse_prediction_rejects_repeated_or_reversed_labels() -> None:
    repeated = parse_prediction("Đáp án: mèo\nGiải thích: Có mèo.\nĐáp án: chó")
    assert repeated.answer == "mèo"
    assert repeated.explanation == "Có mèo."
    assert not repeated.format_valid

    reversed_output = parse_prediction("Giải thích: Có mèo.\nĐáp án: mèo")
    assert reversed_output.answer == "mèo"
    assert not reversed_output.format_valid


def test_parse_prediction_accepts_crlf_and_wrapped_explanation() -> None:
    result = parse_prediction("Đáp án: mèo\r\nGiải thích: Trong ảnh có\r\nmột con mèo.")
    assert result.answer == "mèo"
    assert result.explanation == "Trong ảnh có\r\nmột con mèo."
    assert result.format_valid
