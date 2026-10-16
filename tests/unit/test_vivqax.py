import pytest

from vqa_explanation.data.schema import VQASample
from vqa_explanation.data.vivqax import image_manifest, normalize_record


@pytest.mark.parametrize("field,value", [
    ("question_id", None), ("image_id", None), ("question_id", True),
    ("question_id", {}), ("image_id", 1.5), ("explanations", {"text": "mèo"}),
])
def test_schema_rejects_fake_ids_and_mapping_explanations(field, value):
    row = {
        "question_id": "1001", "image_id": "1", "image_path": "images/1.jpg",
        "question": "Đây là gì?", "answer": "mèo", "explanations": ["Có tai và ria"],
        "split": "train",
    }
    row[field] = value
    with pytest.raises(ValueError):
        VQASample.from_dict(row)


def test_schema_preserves_integer_source_ids():
    row = {
        "question_id": 1001, "image_id": 1, "image_path": "images/1.jpg",
        "question": "Đây là gì?", "answer": "mèo", "explanations": ["Có tai và ria"],
        "split": "train",
    }
    sample = VQASample.from_dict(row)
    assert (sample.question_id, sample.image_id) == ("1001", "1")


def test_normalizer_does_not_turn_null_into_a_valid_id():
    raw = {
        "image_name": "COCO_train2014_000000000001.jpg",
        "question_id": None, "image_id": 1,
    }
    with pytest.raises(ValueError, match="question_id"):
        normalize_record(raw, "train")


def test_normalize_record_maps_official_fields() -> None:
    raw = {
        "question_id": "1001",
        "image_id": "1",
        "image_name": "COCO_train2014_000000000001.jpg",
        "question": "Đây là gì?",
        "answer": "mèo",
        "explanation": ["con vật có tai và ria"],
    }
    result = normalize_record(raw, "train")
    assert result["explanations"] == ["con vật có tai và ria"]
    assert result["image_path"].endswith(raw["image_name"])


def test_image_manifest_merges_repeated_images_and_tracks_splits() -> None:
    records = [
        {
            "image_name": "COCO_val2014_000000000001.jpg",
            "image_id": "1",
            "split": "validation",
        },
        {
            "image_name": "COCO_val2014_000000000001.jpg",
            "image_id": "1",
            "split": "test",
        },
    ]
    result = image_manifest(records)
    assert len(result) == 1
    assert result[0]["dataset_splits"] == "test,validation"
    assert result[0]["coco_split"] == "val2014"


def test_normalize_record_rejects_unsafe_image_name() -> None:
    with pytest.raises(ValueError, match="invalid COCO image_name"):
        normalize_record({"image_name": "../outside.jpg"}, "train")


def test_manifest_rejects_conflicting_image_ids() -> None:
    name = "COCO_train2014_000000000001.jpg"
    with pytest.raises(ValueError, match="multiple image IDs"):
        image_manifest(
            [
                {"image_name": name, "image_id": "1", "split": "train"},
                {"image_name": name, "image_id": "2", "split": "train"},
            ]
        )


def test_manifest_rejects_image_id_reused_for_different_names() -> None:
    with pytest.raises(ValueError, match="image_id maps to multiple"):
        image_manifest(
            [
                {
                    "image_name": "COCO_train2014_000000000001.jpg",
                    "image_id": "1",
                    "split": "train",
                },
                {
                    "image_name": "COCO_train2014_000000000002.jpg",
                    "image_id": "1",
                    "split": "train",
                },
            ]
        )
