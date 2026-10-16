import json
import zipfile
from types import SimpleNamespace

import pytest

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.preference import expansion
from vqa_explanation.preference.quality import assess_reviews


def metadata_fixture(tmp_path):
    old = tmp_path / "old" / "accepted.jsonl"
    write_jsonl(old, [{"fixture": "old"}])
    write_json(old.parent / "manifest.json", {
        "files": {"accepted.jsonl": expansion.sha256_file(old)}
    })
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(candidates, [{"fixture": "candidate"}])
    write_json(tmp_path / "candidates.jsonl.manifest.json", {"fixture": True})
    selection = tmp_path / "pool"
    write_jsonl(selection / "questions.jsonl", [{"fixture": "question"}])
    write_json(selection / "manifest.json", {
        "files": {"questions.jsonl": expansion.sha256_file(selection / "questions.jsonl")}
    })
    return old, candidates, selection


def test_metadata_package_roundtrip_and_immutable_restore(tmp_path):
    old, candidates, selection = metadata_fixture(tmp_path)
    archive = tmp_path / "input.zip"
    result = expansion.package_metadata(old, candidates, selection, archive)
    assert result["images"] == 0 and set(result["files"]) == expansion.METADATA_FILES
    restored = expansion.restore_metadata(archive, result["sha256"], tmp_path / "work")
    for name, digest in result["files"].items():
        assert expansion.sha256_file(restored / name) == digest
    assert expansion.restore_metadata(archive, result["sha256"], tmp_path / "work") == restored
    (restored / "candidates.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata changed"):
        expansion.restore_metadata(archive, result["sha256"], tmp_path / "work")


def test_metadata_restore_rejects_unexpected_members_before_extracting(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.json", "{}")
    with pytest.raises(ValueError, match="members/CRC"):
        expansion.restore_metadata(archive, expansion.sha256_file(archive), tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_metadata_package_refuses_stale_selection_hash(tmp_path):
    old, candidates, selection = metadata_fixture(tmp_path)
    write_jsonl(selection / "questions.jsonl", [{"changed": True}])
    with pytest.raises(ValueError, match="input hash"):
        expansion.package_metadata(old, candidates, selection, tmp_path / "input.zip")
    assert not (tmp_path / "input.zip").exists()


def proposal():
    return {"usable": True, "chosen": "Người bên trái mặc áo màu đỏ.",
            "rejected": "Người bên trái mặc áo màu xanh.", "rejected_span": "màu xanh",
            "evidence": "Áo đỏ ở người phía trái.", "error_type": "attribute"}


def entry():
    return {"sample": {"question_id": "q1", "image_id": "im1", "image_path": "one.jpg",
                       "split": "train", "question": "Người nào cầm gậy?", "answer": "bên trái",
                       "explanations": ["Người mặc áo đỏ."]}, "old_pair": None,
            "candidate": {"raw_output": "Đáp án: bên trái\nGiải thích: Người áo đỏ."}}


def identity():
    return {"recipe_sha256": "a" * 64, "model": {"id": "model", "revision": "c" * 40},
            "adapter": {"sha256": "b" * 64}, "saved_candidate_origin": {"seed": 42},
            "online_candidate_seed": 42, "candidate_manifest_sha256": "d" * 64,
            "candidates_sha256": "e" * 64}


class Backend:
    def __init__(self, *args, fault=None):
        self.visual_calls = 0
        self.fault = fault

    def ask(self, image_path, prompt, tokens):
        if prompt.startswith(expansion.CANDIDATE_REPAIR):
            data = json.loads(prompt[len(expansion.CANDIDATE_REPAIR):])
            assert set(data) == {"fixed_answer", "question", "candidate_explanation"}
            return json.dumps({"usable": False})
        if prompt.startswith(expansion.SYNTHESIS):
            data = json.loads(prompt[len(expansion.SYNTHESIS):])
            assert set(data) == {"fixed_answer", "question"}
            return json.dumps({"usable": True, "chosen": proposal()["chosen"],
                               "correct_span": "màu đỏ", "false_span": "màu xanh",
                               "evidence": proposal()["evidence"], "error_type": "attribute"})
        if prompt.startswith(expansion.TEXT_JUDGE):
            assert image_path is None
            return json.dumps({"semantic_conflict": self.fault == "text",
                               "both_plausible": True,
                               "distinct_visual_facts": self.fault != "equivalent",
                               "answer_entails_preference": self.fault == "entailed",
                               "A_fact": "màu đỏ", "B_fact": "màu xanh",
                               "contrast_reason": "The same shirt cannot have both stated colors.",
                               "reason": "Either shirt color is plausible without the image."})
        self.visual_calls += 1
        return json.dumps({"answer_correct": False if self.fault == "answer" else True,
                           "winner": "A" if self.fault == "order" or self.visual_calls % 2
                           else "B", "both_plausible": True,
                           "evidence": "Áo đỏ ở người phía trái.", "losing_span": "màu xanh",
                           "contrast_relevant": self.fault != "irrelevant",
                           "relevance_reason": "" if self.fault == "rationale"
                           else "The shirt distinguishes the person identified in the question."})


@pytest.mark.parametrize("fault,passed", [(None, True), ("answer", False),
                                         ("order", False), ("text", False),
                                         ("irrelevant", False), ("rationale", False)])
def test_visual_and_text_judgments_are_required(fault, passed):
    ok, raw, parsed = expansion.judge_pair(Backend(fault=fault), entry()["sample"], proposal())
    assert ok is passed
    assert set(raw) == ({"text_only"} if fault == "text"
                        else {"forward", "reverse", "text_only"})
    if fault != "text":
        assert parsed["forward"]["losing_span"] == "màu xanh"


def test_json_failure_retains_raw_response_and_does_not_accept():
    backend = SimpleNamespace(ask=lambda *args: "not JSON")
    ok, raw, parsed = expansion.judge_pair(backend, entry()["sample"], proposal())
    assert not ok and raw["text_only"] == "not JSON" and "error" in parsed


def test_pair_is_tagged_automated_not_human(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"fixture")
    result = expansion.process_row(Backend(), entry(), identity(), "c" * 64, tmp_path)
    assert result["status"] == "added"
    pair, quality = result["pair"], result["quality"]
    assert pair["human_verified"] is False and pair["reviewer_type"] == "ai"
    assert quality["independent_human_verified"] is False
    assert assess_reviews([pair], [quality], 1)["ready"]
    assert pair["provenance"]["chosen"]["kind"] == "synthetic"
    assert pair["provenance"]["original_candidate_origin"]["checkpoint_sha256"] == "b" * 64


def test_no_pair_is_kept_when_text_only_judge_can_choose(tmp_path):
    result = expansion.process_row(Backend(fault="text"), entry(), identity(), "c" * 64, tmp_path)
    assert result["status"] == "excluded" and result["pair"] is None
    assert result["attempts"][-1]["raw"]["text_only"]


def test_saved_m1_error_is_used_before_creating_a_synthetic_negative(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"fixture")
    candidate = entry()
    candidate["candidate"]["raw_output"] = (
        "Đáp án: bên trái\nGiải thích: " + proposal()["rejected"]
    )

    class RepairBackend(Backend):
        def ask(self, image_path, prompt, tokens):
            if prompt.startswith(expansion.CANDIDATE_REPAIR):
                data = json.loads(prompt[len(expansion.CANDIDATE_REPAIR):])
                assert data["candidate_explanation"] == proposal()["rejected"]
                return json.dumps({"usable": True, "wrong_span": "màu xanh",
                                   "correction_span": "màu đỏ", "evidence": proposal()["evidence"]})
            assert not prompt.startswith(expansion.SYNTHESIS)
            return super().ask(image_path, prompt, tokens)

    result = expansion.process_row(RepairBackend(), candidate, identity(), "c" * 64, tmp_path)
    pair = result["pair"]
    assert pair["rejected"] == candidate["candidate"]["raw_output"]
    assert pair["error_type"] == "natural_model_error"
    assert pair["provenance"]["rejected"]["kind"] == "model"
    assert pair["provenance"]["chosen"]["kind"] == "synthetic"
    assert [attempt["method"] for attempt in result["attempts"]] == ["candidate_repair"]
    assert assess_reviews([pair], [result["quality"]], 1, required_version=2)["ready"]


@pytest.mark.parametrize("raw", ["No format", "Đáp án: bên phải\nGiải thích: Áo màu xanh."])
def test_wrong_answer_or_format_is_not_relabelled_as_model_negative(tmp_path, raw):
    (tmp_path / "one.jpg").write_bytes(b"fixture")
    candidate = entry()
    candidate["candidate"]["raw_output"] = raw
    result = expansion.process_row(Backend(), candidate, identity(), "c" * 64, tmp_path)
    assert result["pair"]["provenance"]["rejected"]["kind"] == "synthetic"
    assert result["attempts"][0]["error"] == "M1 answer or format is not reusable"


@pytest.mark.parametrize("change", [{"wrong_span": "missing"}, {"wrong_span": ""},
                                    {"correction_span": "màu xanh"}, {"usable": False}])
def test_model_repair_requires_one_real_edit(change):
    raw = "Đáp án: bên trái\nGiải thích: " + proposal()["rejected"]
    value = {"usable": True, "wrong_span": "màu xanh", "correction_span": "màu đỏ",
             "evidence": proposal()["evidence"]}
    with pytest.raises(ValueError):
        expansion.repair_candidate({**value, **change}, raw, "bên trái")


def test_results_page_has_no_review_names_or_actor_labels(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"fixture")
    result = expansion.process_row(Backend(), entry(), identity(), "c" * 64, tmp_path)
    summary = expansion.write_results(tmp_path, [result], 1000)
    page = (tmp_path / "results.html").read_text("utf-8")
    assert "visual_screen" not in page and "AI" not in page and "reviewer" not in page
    assert "1/1000" in page and "Chosen" in page and summary["accepted"] == 1
    assert list(read_jsonl(tmp_path / "reviewed_pairs.jsonl"))[0]["human_verified"] is False


def test_resume_skips_completed_decisions_and_does_not_load_model_when_done(tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "one.jpg")
    Image.new("RGB", (8, 8), "red").save(tmp_path / "two.jpg")
    sample = entry()
    queue = [sample, {**sample, "sample": {**sample["sample"], "question_id": "q2",
                                          "image_id": "im2", "image_path": "two.jpg"}}]
    monkeypatch.setattr(expansion, "expansion_inputs", lambda *args: (
        queue, {**identity(), "annotations_sha256": "c" * 64}
    ))
    def frozen_fixture(input_path, annotations, image_root, folder):
        folder.mkdir()
        write_jsonl(folder / "accepted.jsonl", list(read_jsonl(input_path)))
        write_jsonl(folder / "chosen_only.jsonl", [{"fixture": True}])
        write_json(folder / "manifest.json", {
            "input_sha256": expansion.sha256_file(input_path), "annotations_sha256": "c" * 64,
            "files": {
            name: expansion.sha256_file(folder / name)
            for name in ("accepted.jsonl", "chosen_only.jsonl")
        }})

    monkeypatch.setattr(expansion, "freeze_pairs", frozen_fixture)
    output = tmp_path / "run"
    args = SimpleNamespace(annotations=tmp_path / "canonical", image_root=tmp_path,
                           old_pairs="old", candidates="candidates", selection="selection",
                           m1_adapter="adapter", m1_selection="m1", output_dir=output,
                           target=2, max_items=1, inspect=False)
    output.mkdir()
    first = expansion.run_expansion(args, backend_factory=Backend)
    assert first["accepted"] == first["processed"] == 1
    second = expansion.run_expansion(args, backend_factory=Backend)
    assert second["accepted"] == second["processed"] == 2
    third = expansion.run_expansion(args, backend_factory=lambda *a: pytest.fail("Unexpected load"))
    assert third["accepted"] == 2
    assert len(list(read_jsonl(output / "decisions.jsonl"))) == 2
    quality_file = output / "frozen" / "quality_review.jsonl"
    changed = list(read_jsonl(quality_file))
    changed[0]["visible_evidence"] = "Changed after freeze"
    write_jsonl(quality_file, changed)
    with pytest.raises(ValueError, match="differ from the decision journal"):
        expansion.run_expansion(args, backend_factory=lambda *a: pytest.fail("Unexpected load"))


@pytest.mark.parametrize("value", ["[]", "true", "```json\n[]\n```"])
def test_judge_output_must_be_an_object(value):
    with pytest.raises(ValueError):
        expansion.json_object(value)


def test_missing_journal_recovers_closed_chunks_and_checks_their_order(tmp_path):
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    path = recovery / "decisions_00000_00002.jsonl"
    rows = [{"question_id": "q1"}, {"question_id": "q2"}]
    write_jsonl(path, rows)
    write_json(path.with_suffix(".sha256.json"), {"sha256": expansion.sha256_file(path),
                                                "first_index": 0, "records": 2})
    assert expansion.restore_decisions(tmp_path) == rows
    assert list(read_jsonl(tmp_path / "decisions.jsonl")) == rows
    assert (tmp_path / "recovery_restore.json").exists()


def test_corrupted_recovery_chunk_cannot_create_a_journal(tmp_path):
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    path = recovery / "decisions_00000_00001.jsonl"
    write_jsonl(path, [{"question_id": "q1"}])
    write_json(path.with_suffix(".sha256.json"), {"sha256": "wrong", "first_index": 0,
                                                "records": 1})
    with pytest.raises(ValueError, match="hash/order"):
        expansion.restore_decisions(tmp_path)
    assert not (tmp_path / "decisions.jsonl").exists()


def test_teacher_is_separate_and_new_candidates_still_use_m1(tmp_path, monkeypatch):
    from vqa_explanation.inference.study_validation import MODEL_ID, MODEL_REVISION

    calls = []

    class Runner:
        def __init__(self, model_id, revision, **kwargs):
            calls.append((model_id, revision, kwargs.get("adapter_path")))
            self.torch = SimpleNamespace(manual_seed=lambda seed: None)

        def generate(self, path, question):
            return "M1 candidate"

    monkeypatch.setattr(expansion, "Qwen25VLRunner", Runner)
    teacher = {"id": "teacher/7b", "revision": "f" * 40}
    backend = expansion.VisualBackend(tmp_path, "m1-adapter", teacher)
    assert calls == [("teacher/7b", "f" * 40, None)]
    assert backend.candidate(entry()["sample"]) == "M1 candidate"
    assert calls[-1] == (MODEL_ID, MODEL_REVISION, "m1-adapter")
    backend.candidate(entry()["sample"])
    assert len(calls) == 2


def test_same_screening_model_shares_m1_without_loading_twice(tmp_path, monkeypatch):
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs["adapter_path"])
            self.torch = SimpleNamespace(manual_seed=lambda seed: None)

        def generate(self, *args):
            return "M1 candidate"

    monkeypatch.setattr(expansion, "Qwen25VLRunner", Runner)
    backend = expansion.VisualBackend(tmp_path, "m1-adapter")
    assert backend.shared_model
    assert backend.candidate(entry()["sample"]) == "M1 candidate"
    assert calls == ["m1-adapter"]


def edit():
    return {"usable": True, "chosen": proposal()["chosen"], "correct_span": "màu đỏ",
            "false_span": "màu xanh", "evidence": proposal()["evidence"],
            "error_type": "attribute"}


def test_negative_is_exactly_one_edit_not_an_unrelated_sentence():
    result = expansion.compose_contrast(edit())
    assert result == proposal()
    assert result["rejected"].replace("màu xanh", "màu đỏ", 1) == result["chosen"]


@pytest.mark.parametrize("change", [
    {"correct_span": "absent"}, {"false_span": "màu đỏ"}, {"false_span": ""},
    {"chosen": "màu đỏ màu đỏ"}, {"error_type": "object/attribute"},
    {"usable": False}, {"chosen": "Đáp án: bên trái\nGiải thích: màu đỏ"},
])
def test_invalid_edit_is_not_silently_repaired(change):
    with pytest.raises(ValueError):
        expansion.compose_contrast({**edit(), **change})


def test_semantic_rejection_skips_expensive_visual_calls():
    backend = Backend(fault="text")
    passed, raw, parsed = expansion.judge_pair(backend, entry()["sample"], proposal())
    assert not passed and backend.visual_calls == 0
    assert set(raw) == {"text_only"} and parsed["text_only"]["semantic_conflict"]


@pytest.mark.parametrize("fault", ["equivalent", "entailed"])
def test_semantically_empty_or_answer_implied_contrast_is_rejected(fault):
    backend = Backend(fault=fault)
    passed, raw, _ = expansion.judge_pair(backend, entry()["sample"], proposal())
    assert not passed and backend.visual_calls == 0 and set(raw) == {"text_only"}


@pytest.mark.parametrize("change", [
    {"distinct_visual_facts": None}, {"answer_entails_preference": None},
    {"A_fact": "missing"}, {"B_fact": "missing"}, {"contrast_reason": ""},
    {"A_fact": "Người bên trái", "B_fact": "Người bên trái"},
])
def test_contrast_gate_requires_auditable_different_facts(change):
    backend = Backend()
    ask = backend.ask

    def response(image_path, prompt, tokens):
        raw = ask(image_path, prompt, tokens)
        if prompt.startswith(expansion.TEXT_JUDGE):
            return json.dumps({**json.loads(raw), **change})
        return raw

    backend.ask = response
    passed, _, _ = expansion.judge_pair(backend, entry()["sample"], proposal())
    assert not passed and backend.visual_calls == 0
