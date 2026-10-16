import json
import sys

import pytest

from scripts.prepare_vivqax import main
from vqa_explanation.data.vivqax import RAW_FILES


def test_prepare_rejects_invalid_annotations_before_writing(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for index, filename in enumerate(RAW_FILES.values(), start=1):
        record = {
            "question_id": "same-question" if index <= 2 else "other-question",
            "image_id": str(index),
            "image_name": f"COCO_train2014_{index:012d}.jpg",
            "question": "Con gì?",
            "answer": "Mèo",
            "explanation": ["Có mèo."],
        }
        (annotations / filename).write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_vivqax.py", "--annotation-dir", str(annotations), "--output-dir", str(output)],
    )
    with pytest.raises(ValueError, match="record/ID audit"):
        main()
    assert not output.exists()


def test_prepare_does_not_overwrite_existing_dataset(tmp_path, monkeypatch):
    output = tmp_path / "processed"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prepare_vivqax.py", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "existing"
