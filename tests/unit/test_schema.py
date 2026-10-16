import pytest

from vqa_explanation.data.schema import VQASample


def test_schema_accepts_single_explanation() -> None:
    sample = VQASample.from_dict(
        {
            "question_id": "q1",
            "image_id": "1",
            "image_path": "image.jpg",
            "question": "Con vật nào xuất hiện trong ảnh?",
            "answer": "Con mèo",
            "explanation": "Ảnh cho thấy một con mèo trên ghế.",
            "split": "val",
        }
    )
    assert sample.split == "validation"
    assert len(sample.explanations) == 1


def test_schema_rejects_empty_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        VQASample.from_dict(
            {
                "question_id": "q1",
                "image_id": "1",
                "image_path": "image.jpg",
                "question": "Câu hỏi",
                "answer": " ",
                "explanation": "Giải thích",
                "split": "train",
            }
        )

