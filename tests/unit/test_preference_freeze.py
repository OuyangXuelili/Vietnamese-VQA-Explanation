"""Frozen preference exports must remain train-only and share the exact chosen target."""

import copy
import json

import pytest

from scripts import validate_preferences
from vqa_explanation.data.io import read_jsonl, write_jsonl
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.preference.freeze import freeze_pairs, prepare_exports
from vqa_explanation.training.sft_data import target_for
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def inputs(tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(tmp_path / "a.jpg")
    canonical = [{"question_id": "q1", "image_id": "i1", "image_path": "a.jpg",
                  "question": "Con gì?", "answer": "mèo", "explanations": ["Có mèo."],
                  "split": "train"}]
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, canonical)
    pair = {"pair_id": "p1", "source_question_id": "q1", "image_path": "a.jpg",
            "prompt": build_user_prompt("Con gì?"),
            "chosen": "Đáp án: mèo\nGiải thích: Có mèo màu đỏ.",
            "rejected": "Đáp án: mèo\nGiải thích: Có mèo màu xanh.",
            "error_type": "attribute", "generation_method": "manual",
            "review_status": "accepted", "reviewer": "test-fixture-not-real-review",
            "reviewed_at": "2026-09-28T12:00:00+07:00", "evidence": "test only",
            "rejected_span": "màu xanh", "annotations_sha256": sha256_file(annotations),
            "image_sha256": sha256_file(tmp_path / "a.jpg"),
            "provenance": {"chosen": {"kind": "reference", "source": "fixture"},
                           "rejected": {"kind": "manual", "source": "fixture"}}}
    return tmp_path, canonical, annotations, pair


def test_freeze_exports_same_chosen_and_refuses_overwrite(inputs):
    root, canonical, annotations, pair = inputs
    candidate_path = root / "pairs.jsonl"
    pending = {**pair, "pair_id": "pending", "review_status": "pending"}
    write_jsonl(candidate_path, [pair, pending])
    out = root / "frozen"
    manifest = freeze_pairs(candidate_path, annotations, root, out)
    accepted = list(read_jsonl(out / "accepted.jsonl"))
    chosen = list(read_jsonl(out / "chosen_only.jsonl"))
    assert len(accepted) == len(chosen) == manifest["pairs"] == 1
    assert accepted[0]["chosen"] == target_for(chosen[0])
    assert accepted[0]["provenance"] == pair["provenance"]
    assert manifest["review_counts"]["review_pending"] == 1
    for name, digest in manifest["files"].items():
        assert sha256_file(out / name) == digest
    with pytest.raises(FileExistsError):
        freeze_pairs(candidate_path, annotations, root, out)


def ai_pair(pair):
    item = copy.deepcopy(pair)
    item.update(reviewer_type="ai", human_verified=False, generation_method="ai_review_candidate")
    for side in ("chosen", "rejected"):
        item["provenance"][side] = {
            "kind": "synthetic", "source": "recipe:q1", "recipe_sha256": "d" * 64,
        }
    return item


def test_freeze_records_ai_review_separately(inputs):
    root, _, annotations, pair = inputs
    path = root / "pairs.jsonl"
    write_jsonl(path, [ai_pair(pair)])
    manifest = freeze_pairs(path, annotations, root, root / "frozen")
    assert manifest["reviewer_types"] == {"ai": 1}
    assert manifest["human_verified_pairs"] == 0


@pytest.mark.parametrize("field,value", [
    ("human_verified", True), ("human_verified", 0),
    ("generation_method", "human_review_candidate"), ("reviewer_type", "human"),
])
def test_freeze_rejects_false_review_actor_claims(inputs, field, value):
    root, canonical, annotations, pair = inputs
    row = ai_pair(pair)
    row[field] = value
    with pytest.raises(ValueError, match="human_verified|generation_method"):
        prepare_exports([row], canonical, root, sha256_file(annotations))


def test_ai_pair_cannot_hide_manual_origin_at_freeze(inputs):
    root, canonical, annotations, pair = inputs
    row = ai_pair(pair)
    row["provenance"]["chosen"] = {"kind": "manual", "source": "invented person"}
    with pytest.raises(ValueError, match="manual origin"):
        prepare_exports([row], canonical, root, sha256_file(annotations))


def test_ai_pair_requires_explicit_nonhuman_flag(inputs):
    root, canonical, annotations, pair = inputs
    row = ai_pair(pair)
    del row["human_verified"]
    with pytest.raises(ValueError, match="human_verified=false"):
        prepare_exports([row], canonical, root, sha256_file(annotations))


@pytest.mark.parametrize(("field", "value", "error"), [
    ("source_question_id", "test-id", "canonical train"),
    ("image_path", "other.jpg", "image_path"),
    ("prompt", "wrong question", "prompt"),
    ("image_sha256", "0" * 64, "image_sha256"),
    ("annotations_sha256", "0" * 64, "annotations_sha256"),
    ("reviewer", "", "reviewer"),
    ("reviewed_at", "2026-09-28", "timezone"),
    ("evidence", None, "evidence"),
    ("rejected_span", "không có", "rejected_span"),
    ("provenance", {}, "provenance.chosen"),
    ("chosen", "mèo", "format"),
    ("rejected", "Đáp án: chó\nGiải thích: màu xanh", "answers differ"),
    ("rejected", "Đáp án: mèo\nGiải thích: có mèo MÀU đỏ.", "explanations must differ"),
])
def test_invalid_pairs_never_create_export(inputs, field, value, error):
    root, _, annotations, pair = inputs
    pair[field] = value
    candidate_path = root / "pairs.jsonl"
    write_jsonl(candidate_path, [pair])
    with pytest.raises(ValueError, match=error):
        freeze_pairs(candidate_path, annotations, root, root / "frozen")
    assert not (root / "frozen").exists()


def test_rejects_heldout_and_image_overlap(inputs):
    root, canonical, annotations, pair = inputs
    canonical[0]["split"] = "test"
    with pytest.raises(ValueError, match="canonical train"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))
    canonical[0]["split"] = "train"
    heldout = {**canonical[0], "question_id": "q2", "split": "validation"}
    with pytest.raises(ValueError, match="crosses splits"):
        prepare_exports([pair], canonical + [heldout], root, sha256_file(annotations))
    heldout["image_id"] = "different-id"
    with pytest.raises(ValueError, match="crosses splits"):
        prepare_exports([pair], canonical + [heldout], root, sha256_file(annotations))


def test_rejects_duplicate_question_and_content(inputs):
    root, canonical, annotations, pair = inputs
    duplicate = {**pair, "pair_id": "p2"}
    with pytest.raises(ValueError, match="Duplicate accepted source"):
        prepare_exports([pair, duplicate], canonical, root, sha256_file(annotations))
    canonical.append({**canonical[0], "question_id": "q2"})
    duplicate["source_question_id"] = "q2"
    with pytest.raises(ValueError, match="duplicate accepted content"):
        prepare_exports([pair, duplicate], canonical, root, sha256_file(annotations))


def test_never_promotes_pending_to_human_accepted(inputs):
    root, canonical, annotations, pair = inputs
    pair["review_status"] = "pending"
    original = copy.deepcopy(pair)
    with pytest.raises(ValueError, match="No accepted pairs"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))
    assert pair == original


@pytest.mark.parametrize("kind", [[], {}, None, 1])
def test_invalid_provenance_kind_is_validation_error(inputs, kind):
    root, _, annotations, pair = inputs
    pair["provenance"]["rejected"]["kind"] = kind
    candidate_path = root / "pairs.jsonl"
    write_jsonl(candidate_path, [pair])
    with pytest.raises(ValueError, match="provenance.rejected.kind"):
        freeze_pairs(candidate_path, annotations, root, root / "frozen")
    assert not (root / "frozen").exists()


def test_export_canonicalizes_paths_and_rejects_same_wrong_answer(inputs):
    root, canonical, annotations, pair = inputs
    pair["image_path"] = " a.jpg "
    accepted, chosen, _ = prepare_exports([pair], canonical, root, sha256_file(annotations))
    assert accepted[0]["image_path"] == chosen[0]["image_path"] == "a.jpg"
    pair["chosen"] = pair["chosen"].replace("Đáp án: mèo", "Đáp án: chó")
    pair["rejected"] = pair["rejected"].replace("Đáp án: mèo", "Đáp án: chó")
    with pytest.raises(ValueError, match="answers differ"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))


def test_model_provenance_requires_checkpoint_recipe_and_seed(inputs):
    root, canonical, annotations, pair = inputs
    origin = {"kind": "model", "source": "M1"}
    pair["provenance"]["rejected"] = origin
    with pytest.raises(ValueError, match="recipe_sha256"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))
    origin.update(recipe_sha256="a" * 64, model_id="fixture", checkpoint_sha256="b" * 64,
                  seed=True)
    with pytest.raises(ValueError, match="seed"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))
    origin["seed"] = 42
    assert len(prepare_exports([pair], canonical, root, sha256_file(annotations))[0]) == 1


@pytest.mark.parametrize("relative", ["../a.jpg", "/a.jpg", "a/../a.jpg", "a\\b.jpg", "C:/a.jpg"])
def test_rejects_nonportable_paths(inputs, relative):
    root, canonical, annotations, pair = inputs
    canonical[0]["image_path"] = relative
    with pytest.raises(ValueError, match="image path"):
        prepare_exports([pair], canonical, root, sha256_file(annotations))


def test_cli_freeze(inputs, monkeypatch, capsys):
    root, _, annotations, pair = inputs
    write_jsonl(root / "pairs.jsonl", [pair])
    monkeypatch.setattr("sys.argv", [
        "validate_preferences.py", "--input", str(root / "pairs.jsonl"),
        "--annotations", str(annotations), "--image-root", str(root),
        "--output-dir", str(root / "frozen"),
    ])
    validate_preferences.main()
    assert json.loads(capsys.readouterr().out)["pairs"] == 1
