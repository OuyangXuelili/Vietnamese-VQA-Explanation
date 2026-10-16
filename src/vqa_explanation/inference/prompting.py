from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "Bạn là hệ thống hỏi đáp hình ảnh tiếng Việt. "
    "Chỉ sử dụng thông tin quan sát được trong ảnh."
)


def build_user_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    return (
        f"Câu hỏi: {question}\n"
        "Trả lời đúng hai phần theo định dạng:\n"
        "Đáp án: <câu trả lời ngắn>\n"
        "Giải thích: <một câu nêu bằng chứng nhìn thấy trong ảnh>"
    )


@dataclass(frozen=True)
class ParsedPrediction:
    answer: str
    explanation: str
    format_valid: bool


def parse_prediction(text: str) -> ParsedPrediction:
    cleaned = text.strip()
    answer_pattern = r"^[ \t]*Đáp[ \t]*án[ \t]*:[ \t]*([^\r\n]*)"
    explanation_pattern = r"^[ \t]*Giải[ \t]*thích[ \t]*:[ \t]*([^\r\n]*)"
    answer_matches = list(re.finditer(answer_pattern, cleaned, flags=re.IGNORECASE | re.MULTILINE))
    explanation_matches = list(
        re.finditer(explanation_pattern, cleaned, flags=re.IGNORECASE | re.MULTILINE)
    )
    answer = answer_matches[0].group(1).strip() if answer_matches else ""
    explanation = ""
    if explanation_matches:
        start = explanation_matches[0].start(1)
        following_labels = [
            match.start()
            for match in (*answer_matches, *explanation_matches)
            if match.start() > explanation_matches[0].start()
        ]
        explanation = cleaned[start : min(following_labels, default=len(cleaned))].strip()
    format_pattern = (
        r"\A[ \t]*Đáp[ \t]*án[ \t]*:[ \t]*([^\r\n]*)\r?\n"
        r"[ \t]*Giải[ \t]*thích[ \t]*:[ \t]*(.*?)\Z"
    )
    expected_format = re.fullmatch(format_pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
    return ParsedPrediction(
        answer=answer,
        explanation=explanation,
        format_valid=bool(
            expected_format
            and len(answer_matches) == 1
            and len(explanation_matches) == 1
            and answer
            and explanation
        ),
    )
