import json

import pytest

from vqa_explanation.data.io import read_jsonl, write_jsonl
from vqa_explanation.preference.selection import freeze_selection, select_questions
from vqa_explanation.utils.run_provenance import sha256_file


def records():
    rows = [{"question_id": f"q{i}", "image_id": f"i{i}", "image_path": f"{i}.jpg",
             "question": "Con gì?", "answer": "mèo", "explanations": ["Có mèo."],
             "split": "train"} for i in range(4)]
    return rows + [{**rows[0], "question_id": "q4"},
                   {**rows[0], "question_id": "v", "image_id": "v",
                    "image_path": "v.jpg", "split": "validation"},
                   {**rows[0], "question_id": "t", "image_id": "t",
                    "image_path": "t.jpg", "split": "test"}]


def test_selection_is_order_independent_train_only_one_per_image(tmp_path):
    rows = records()
    chosen = select_questions(rows, tmp_path, 4)
    assert chosen == select_questions(rows[::-1], tmp_path, 4)
    assert len({r["image_id"] for r in chosen}) == 4
    assert all(r["split"] == "train" and r["review_status"] == "pending" for r in chosen)
    assert not any("chosen" in r or "rejected" in r for r in chosen)


@pytest.mark.parametrize("change", ["overlap", "alias", "duplicate", "escape", "train_only"])
def test_selection_rejects_invalid_canonical_data(tmp_path, change):
    rows = records()
    if change == "overlap":
        rows[-1]["image_id"] = "i0"
    elif change == "alias":
        rows[-1]["image_path"] = "sub/../0.jpg"
    elif change == "duplicate":
        rows[-1]["question_id"] = "q0"
    elif change == "escape":
        rows[0]["image_path"] = "../outside.jpg"
    else:
        rows = rows[:4]
    with pytest.raises(ValueError):
        select_questions(rows, tmp_path, 2)


@pytest.mark.parametrize("count", [0, 5, 2001, True])
def test_selection_never_silently_reduces_count(tmp_path, count):
    with pytest.raises(ValueError):
        select_questions(records(), tmp_path, count)


def test_freeze_checks_files_before_export_and_never_overwrites(tmp_path):
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, records())
    output = tmp_path / "selection"
    with pytest.raises(FileNotFoundError):
        freeze_selection(annotations, tmp_path, output, 4)
    assert not output.exists()
    for i in range(4):
        (tmp_path / f"{i}.jpg").write_bytes(f"fixture {i}".encode())
    manifest = freeze_selection(annotations, tmp_path, output, 4)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["files"]["questions.jsonl"] == sha256_file(output / "questions.jsonl")
    for row in read_jsonl(output / "questions.jsonl"):
        assert row["image_sha256"] == sha256_file(tmp_path / row["image_path"])
        assert row["annotations_sha256"] == sha256_file(annotations)
    with pytest.raises(FileExistsError):
        freeze_selection(annotations, tmp_path, output, 4)


def test_new_pool_excludes_all_questions_from_old_images(tmp_path):
    rows = records()
    excluded = [{"source_question_id": "q0", "image_id": "i0", "image_path": "0.jpg",
                 "split": "train"}]
    selected = select_questions(rows, tmp_path, 3, excluded=excluded)
    assert not any(row["image_id"] == "i0" for row in selected)
    assert {r["source_question_id"] for r in selected} == {"q1", "q2", "q3"}
    with pytest.raises(ValueError, match="Only 3"):
        select_questions(rows, tmp_path, 4, excluded=excluded)


def test_selection_preserves_canonical_whitespace_for_candidate_export(tmp_path):
    from vqa_explanation.preference.candidates import candidate_records

    rows = [row for row in records() if row["question_id"] != "q4"]
    rows[0]["answer"] = "mèo "
    rows[0]["question"] = " Con gì? "
    rows[0]["explanations"] = [" Có mèo. "]
    annotations, output = tmp_path / "all.jsonl", tmp_path / "pool"
    write_jsonl(annotations, rows)
    for i in range(4):
        (tmp_path / f"{i}.jpg").write_bytes(b"fixture")
    freeze_selection(annotations, tmp_path, output, 4)
    selected, _ = candidate_records(output, annotations, tmp_path, sha256_file(annotations), 4)
    assert next(r for r in selected if r["question_id"] == "q0")["answer"] == "mèo "


def test_freeze_exclusions_verifies_manifests_before_writing(tmp_path):
    rows = records()
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, rows)
    for i in range(4):
        (tmp_path / f"{i}.jpg").write_bytes(b"fixture")
    old, new = tmp_path / "old", tmp_path / "new"
    freeze_selection(annotations, tmp_path, old, 2)
    result = freeze_selection(annotations, tmp_path, new, 2, exclude_selections=[old])
    assert result["excluded_images"] == 2
    old_images = {r["image_id"] for r in read_jsonl(old / "questions.jsonl")}
    assert old_images.isdisjoint(r["image_id"] for r in read_jsonl(new / "questions.jsonl"))
    (old / "questions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest/hash"):
        freeze_selection(annotations, tmp_path, tmp_path / "invalid", 1,
                         exclude_selections=[old])
    assert not (tmp_path / "invalid").exists()
