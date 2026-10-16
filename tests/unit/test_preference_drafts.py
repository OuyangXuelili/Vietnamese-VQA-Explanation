import copy
import json

import pytest

from scripts.attach_preference_drafts import attach_drafts
from scripts.build_preference_pilot import build_pilot
from vqa_explanation.data.io import write_json, write_jsonl
from vqa_explanation.preference.drafts import load_review_drafts
from vqa_explanation.preference.review import blank_notes, render_editor
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def draft_case(tmp_path):
    image = tmp_path / "images/one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image fixture")
    inputs = [dict(question_id="q1", image_path="images/one.jpg", question="Animal?",
                   answer="cat", split="train")]
    candidates = [dict(question_id="q1", prediction_answer="cat", prediction_explanation="Fur.")]
    write_jsonl(tmp_path / "inputs.jsonl", inputs)
    write_jsonl(tmp_path / "candidates.jsonl", candidates)
    write_json(tmp_path / "candidates.jsonl.manifest.json", {
        "inputs": {"context": {"annotations_sha256": "a" * 64},
                   "image_hashes": {"q1": sha256_file(image)}}})
    notes = blank_notes(inputs, {"q1": candidates[0]}, "b" * 64)
    write_jsonl(tmp_path / "review_notes.jsonl", notes)
    (tmp_path / "index.html").write_text(render_editor(notes) + '<article>Keep</article>',
                                         encoding="utf-8")
    recipe = {
        "source_bundle_sha256": "b" * 64, "annotations_sha256": "a" * 64,
        "selection": {"question_ids": ["q1"]},
        "transformations": [{"source_question_id": "q1", "chosen_explanation": "Has fur.",
                             "replace_from": "fur", "replace_to": "wings",
                             "error_type": "attribute"}],
    }
    for key, name in (("source_inputs", "inputs.jsonl"),
                      ("source_candidates", "candidates.jsonl"),
                      ("source_candidate_manifest", "candidates.jsonl.manifest.json")):
        recipe[key] = {"sha256": sha256_file(tmp_path / name)}
    write_json(tmp_path / "recipe.json", recipe)
    origin = {"kind": "synthetic", "source": "fixture:draft-q1",
              "recipe_sha256": sha256_file(tmp_path / "recipe.json")}
    draft = {
        "schema": "preference_ai_draft_v1", "source_question_id": "q1",
        "status": "pending_ai_draft", "human_verified": False, "reviewer": None,
        "reviewed_at": None, "split": "train", "question": "Animal?", "answer": "cat",
        "original_candidate_answer": "cat", "original_candidate_explanation": "Fur.",
        "canonical_image_path": "images/one.jpg", "image_sha256": sha256_file(image),
        "annotations_sha256": "a" * 64, "recipe_sha256": origin["recipe_sha256"],
        "source_candidate": {"question_id": "q1", **recipe["source_candidates"]},
        "proposed_chosen_explanation": "Has fur.", "proposed_rejected_explanation": "Has wings.",
        "unsupported_span": "wings", "image_evidence_ai": "Visible fur.",
        "error_type": "attribute", "origin_kind": "synthetic",
        "chosen_provenance": origin, "rejected_provenance": copy.deepcopy(origin),
    }
    write_jsonl(tmp_path / "drafts.jsonl", [draft])
    return tmp_path, draft


def load(case):
    root, _ = case
    return load_review_drafts(root, root / "drafts.jsonl", root / "recipe.json")


def test_attach_preserves_notes_images_and_original_page(draft_case):
    root, _ = draft_case
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = attach_drafts(root, root / "drafts.jsonl", root / "recipe.json")
    assert result["ai_drafts"] == 1 and result["accepted"] == 0
    assert result["images_copied"] == 0
    assert all(p.read_bytes() == data for p, data in before.items())
    page = (root / "draft_review.html").read_text(encoding="utf-8")
    notes = json.loads(page.split('id="review-data">')[1].split('</script>')[0])
    assert notes[0]["reviewer"] is None and notes[0]["review_status"] == "pending"
    assert notes[0]["proposed_chosen_explanation"] is None
    assert page.endswith('<article>Keep</article>')
    with pytest.raises(FileExistsError):
        attach_drafts(root, root / "drafts.jsonl", root / "recipe.json")


@pytest.mark.parametrize("field,value", [
    ("human_verified", True), ("reviewer", "invented"), ("status", "accepted"),
    ("split", "test"), ("answer", "dog"), ("image_sha256", "0" * 64),
    ("recipe_sha256", "0" * 64), ("original_candidate_explanation", "Changed"),
    ("canonical_image_path", "../outside.jpg"), ("unsupported_span", "absent"),
    ("proposed_rejected_explanation", "Has many wings."), ("origin_kind", "manual"),
])
def test_rejects_unverified_or_changed_sources(draft_case, field, value):
    root, draft = draft_case
    draft[field] = value
    write_jsonl(root / "drafts.jsonl", [draft])
    with pytest.raises(ValueError):
        load(draft_case)


def test_rejects_duplicate_ids(draft_case):
    root, draft = draft_case
    write_jsonl(root / "drafts.jsonl", [draft, draft])
    with pytest.raises(ValueError, match="IDs"):
        load(draft_case)


def test_changed_source_cannot_be_imported(draft_case):
    root, _ = draft_case
    with (root / "candidates.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load(draft_case)


def test_existing_human_edits_are_not_replaced(draft_case):
    root, _ = draft_case
    notes, drafts = load(draft_case)
    notes[0]["image_evidence"] = "Human note, keep it."
    write_jsonl(root / "review_notes.jsonl", notes)
    loaded, _ = load(draft_case)
    assert loaded == notes
    page = render_editor(loaded, drafts=drafts)
    assert "apply.disabled=!isBlank(row)" in page
    assert "if (!draft || !isBlank(row)) return" in page


def test_draft_script_text_is_escaped(draft_case):
    notes, drafts = load(draft_case)
    drafts[0]["image_evidence_ai"] = "</script><script>alert(1)</script>"
    page = render_editor(notes, drafts=drafts)
    assert "</script><script>alert(1)" not in page
    assert "\\u003c/script>" in page


def test_refresh_only_updates_an_unmodified_generated_page(draft_case):
    root, _ = draft_case
    attach_drafts(root, root / "drafts.jsonl", root / "recipe.json")
    attach_drafts(root, root / "drafts.jsonl", root / "recipe.json", refresh=True)
    with (root / "draft_review.html").open("a", encoding="utf-8") as stream:
        stream.write("User edit")
    with pytest.raises(ValueError, match="edited"):
        attach_drafts(root, root / "drafts.jsonl", root / "recipe.json", refresh=True)


def test_editor_waits_for_articles_before_initial_draw(draft_case):
    notes, drafts = load(draft_case)
    page = render_editor(notes, drafts=drafts)
    assert "document.addEventListener('DOMContentLoaded',draw)" in page
    assert 'id="review-context"' in page


def test_build_pilot_uses_fixed_prefix_and_keeps_all_notes(draft_case):
    root, _ = draft_case
    before = (root / "review_notes.jsonl").read_bytes()
    result = build_pilot(root, root / "drafts.jsonl", root / "recipe.json", count=1)
    assert result["question_ids"] == ["q1"]
    assert result["unique_images"] == 1
    assert result["candidate_answer_exact_match"] == 1
    assert result["review_status_counts"] == {
        "pending": 1, "accepted": 0, "rejected": 0, "uncertain": 0
    }
    assert (root / "review_notes.jsonl").read_bytes() == before
    page = (root / "pilot_review_1.html").read_text(encoding="utf-8")
    queue = json.loads(page.split('id="queue-data">')[1].split("</script>")[0])
    assert queue["question_ids"] == ["q1"]
    assert 'id="next-pending"' in page and 'id="queue-progress"' in page


def test_pilot_refresh_refuses_to_replace_edited_page(draft_case):
    root, _ = draft_case
    build_pilot(root, root / "drafts.jsonl", root / "recipe.json", count=1)
    build_pilot(root, root / "drafts.jsonl", root / "recipe.json", count=1, refresh=True)
    with (root / "pilot_review_1.html").open("a", encoding="utf-8") as stream:
        stream.write("User edit")
    with pytest.raises(ValueError, match="edited"):
        build_pilot(root, root / "drafts.jsonl", root / "recipe.json", count=1,
                    refresh=True)


@pytest.mark.parametrize("queue", [
    {"schema": "candidate_review_queue_v1", "label": "Pilot", "question_ids": []},
    {"schema": "candidate_review_queue_v1", "label": "Pilot", "question_ids": ["q1", "q1"]},
    {"schema": "candidate_review_queue_v1", "label": "Pilot", "question_ids": ["missing"]},
])
def test_editor_rejects_invalid_queue(draft_case, queue):
    notes, drafts = load(draft_case)
    with pytest.raises(ValueError, match="queue"):
        render_editor(notes, drafts=drafts, queue=queue)
