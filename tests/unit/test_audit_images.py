from PIL import Image

from vqa_explanation.data.audit import audit_records


def row(path):
    return {
        "question_id": "q1",
        "image_id": "1",
        "image_path": path,
        "question": "Con gì?",
        "answer": "Mèo",
        "explanations": ["Có mèo."],
        "split": "train",
    }


def test_audit_detects_corrupt_file(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    report = audit_records([row("bad.jpg")], True, tmp_path)
    assert report["counts"]["corrupt_images"] == 1


def test_audit_accepts_valid_image(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "ok.jpg")
    report = audit_records([row("ok.jpg")], True, tmp_path)
    assert report["counts"].get("corrupt_images", 0) == 0


def test_audit_blocks_path_escape(tmp_path):
    report = audit_records([row("../outside.jpg")], True, tmp_path)
    assert report["counts"]["unsafe_image_paths"] == 1


def test_audit_reports_non_object_record_without_crashing():
    report = audit_records([[]])
    assert report["counts"]["invalid"] == 1
    assert "JSON object" in report["errors"][0]["error"]


def test_audit_finds_path_overlap_even_with_different_image_ids():
    left = row("images/COCO_train2014_000000000001.jpg")
    right = {**left, "question_id": "q2", "image_id": "2", "split": "test"}
    report = audit_records([left, right])
    assert report["image_overlap_by_split"]["test__train"] == 0
    assert report["image_path_overlap_by_split"]["test__train"] == 1


def test_audit_finds_equivalent_paths_across_splits(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "ok.jpg")
    left = row("images/ok.jpg")
    right = {**left, "question_id": "q2", "image_id": "2", "split": "test",
             "image_path": "images/../images/ok.jpg"}
    report = audit_records([left, right], True, tmp_path)
    assert report["image_overlap_by_split"]["test__train"] == 0
    assert report["image_path_overlap_by_split"]["test__train"] == 1
