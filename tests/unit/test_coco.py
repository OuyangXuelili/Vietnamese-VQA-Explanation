from pathlib import Path

from PIL import Image

from vqa_explanation.data.coco import coco_image_url, select_smoke_images, valid_jpeg


def test_select_smoke_images_is_deterministic_and_excludes_overlap() -> None:
    rows = [
        {
            "image_name": f"COCO_train2014_{index:012d}.jpg",
            "image_id": str(index),
            "coco_split": "train2014",
            "dataset_splits": "train",
        }
        for index in range(10)
    ]
    rows += [
        {
            "image_name": "COCO_val2014_000000000100.jpg",
            "image_id": "100",
            "coco_split": "val2014",
            "dataset_splits": "test,validation",
        }
    ]
    first = select_smoke_images(rows, {"train": 4}, seed=7)
    second = select_smoke_images(rows, {"train": 4}, seed=7)
    assert first == second
    assert len(first) == 4
    assert all(row["dataset_splits"] == "train" for row in first)


def test_coco_image_url_uses_source_split() -> None:
    row = {
        "image_name": "COCO_val2014_000000000001.jpg",
        "coco_split": "val2014",
    }
    assert coco_image_url(row).endswith("/val2014/COCO_val2014_000000000001.jpg")


def test_valid_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4), color="red").save(image_path, format="JPEG")
    assert valid_jpeg(image_path)
