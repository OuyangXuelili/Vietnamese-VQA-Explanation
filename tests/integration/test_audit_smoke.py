from vqa_explanation.data.audit import audit_records


def test_audit_smoke() -> None:
    rows = [
        {
            "question_id": "q1",
            "image_id": "1",
            "image_path": "image.jpg",
            "question": "Đây là con gì?",
            "answer": "Con mèo",
            "explanations": ["Con vật trong ảnh có hình dáng của mèo."],
            "split": "train",
        }
    ]
    result = audit_records(rows)
    assert result["counts"]["records"] == 1
    assert result["counts"]["split_train"] == 1

