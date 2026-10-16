ANSWER_TAG = "Đáp án:"
EXPLANATION_TAG = "Giải thích:"


def format_target(answer: str, explanation: str) -> str:
    return f"{ANSWER_TAG} {answer.strip()}\n{EXPLANATION_TAG} {explanation.strip()}"


def format_prompt(question: str) -> str:
    return (
        "Hãy trả lời câu hỏi dựa trên nội dung ảnh và giải thích ngắn gọn bằng tiếng Việt.\n"
        f"Câu hỏi: {question.strip()}"
    )

