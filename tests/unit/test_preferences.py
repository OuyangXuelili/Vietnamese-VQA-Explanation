import json

import pytest

from scripts import validate_preferences
from vqa_explanation.preference.schema import PreferencePair
from vqa_explanation.preference.validate import validate_pairs


def sample_pair(pair_id="p1"):
    return {
        "pair_id": pair_id,
        "source_question_id": "q1",
        "image_path": "image.jpg",
        "prompt": "Câu hỏi",
        "chosen": "Đáp án: mèo\nGiải thích: có một con mèo.",
        "rejected": "Đáp án: chó\nGiải thích: có một con chó.",
        "error_type": "object",
        "generation_method": "manual",
        "review_status": "accepted",
    }


def test_preference_pair_rejects_identical_outputs() -> None:
    raw = sample_pair()
    raw["rejected"] = raw["chosen"]
    with pytest.raises(ValueError, match="must be different"):
        PreferencePair.from_dict(raw)


@pytest.mark.parametrize(
    "raw", [None, [], {**sample_pair(), "chosen": None}, {**sample_pair(), "pair_id": 7}]
)
def test_preference_pair_rejects_non_object_or_non_string_fields(raw):
    with pytest.raises((TypeError, ValueError)):
        PreferencePair.from_dict(raw)


def test_duplicate_pair_ids_fail_validation_and_cli(tmp_path, monkeypatch, capsys):
    rows = [sample_pair(), sample_pair()]
    report = validate_pairs(rows)
    assert report["counts"]["invalid"] == 1
    assert report["counts"]["duplicate_pair_ids"] == 1
    assert len(report["errors"]) == 1

    path = tmp_path / "duplicates.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_preferences.py", "--input", str(path)])
    with pytest.raises(SystemExit, match="1"):
        validate_preferences.main()
    assert "duplicate_pair_ids" in capsys.readouterr().out


def test_empty_preference_input_is_not_a_valid_dataset():
    report = validate_pairs([])
    assert report["counts"] == {"invalid": 1, "records": 0}
    assert report["errors"][0]["error"] == "no preference pairs"
