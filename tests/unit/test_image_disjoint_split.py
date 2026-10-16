import pytest

from scripts.make_image_disjoint_split import derive_image_disjoint


def row(split, image_id, question_id):
    return {
        "split": split,
        "image_id": image_id,
        "image_name": f"COCO_val2014_{int(image_id):012d}.jpg",
        "image_path": f"images/COCO_val2014_{int(image_id):012d}.jpg",
        "question_id": question_id,
        "question": "Có gì?",
        "answer": "Một vật.",
        "explanations": ["Nhìn thấy một vật."],
    }


def test_preserves_train_and_test_drops_shared_validation_image():
    records = [
        row("train", "1", "q1"),
        row("validation", "2", "q2"),
        row("validation", "3", "q3"),
        row("validation", "3", "q4"),
        row("test", "3", "q5"),
    ]
    result, shared = derive_image_disjoint(records)
    assert shared == ["3"]
    assert {r["question_id"] for r in result} == {"q1", "q2", "q5"}


def test_refuses_train_leakage():
    records = [row("train", "1", "q1"), row("validation", "1", "q2"), row("test", "2", "q3")]
    with pytest.raises(ValueError, match="Train overlaps"):
        derive_image_disjoint(records)


def test_refuses_duplicate_questions():
    records = [
        row("train", "1", "q1"),
        row("validation", "2", "q1"),
        row("test", "3", "q3"),
    ]
    with pytest.raises(ValueError, match="audit"):
        derive_image_disjoint(records)


def test_refuses_inconsistent_image_identity():
    records = [
        row("train", "1", "q1"),
        row("validation", "2", "q2"),
        row("test", "3", "q3"),
    ]
    records[1]["image_name"] = records[0]["image_name"]
    records[1]["image_path"] = records[0]["image_path"]
    with pytest.raises(ValueError, match="filename maps to multiple IDs"):
        derive_image_disjoint(records)


def test_refuses_alias_split_that_would_be_omitted_from_output():
    records = [row("train", "1", "q1"), row("validation", "2", "q2"), row("test", "3", "q3")]
    records[1]["split"] = "val"
    with pytest.raises(ValueError, match="canonical train/validation/test"):
        derive_image_disjoint(records)


def test_refuses_image_path_that_does_not_match_image_name():
    records = [row("train", "1", "q1"), row("validation", "2", "q2"), row("test", "3", "q3")]
    records[1]["image_path"] = records[0]["image_path"]
    with pytest.raises(ValueError, match="image_path must match"):
        derive_image_disjoint(records)
