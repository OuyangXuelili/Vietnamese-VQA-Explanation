import pytest

from vqa_explanation.data.splits import assert_no_image_leakage


def test_detects_image_leakage() -> None:
    rows = [
        {"image_id": "7", "split": "train"},
        {"image_id": "7", "split": "test"},
    ]
    with pytest.raises(ValueError, match="image leakage"):
        assert_no_image_leakage(rows)


def test_allows_disjoint_images() -> None:
    rows = [
        {"image_id": "7", "split": "train"},
        {"image_id": "8", "split": "test"},
    ]
    assert_no_image_leakage(rows)

