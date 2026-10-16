import copy
import hashlib
import io
import json
import sys
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import finalize_preference_review
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.preference.handoff import build_handoff
from vqa_explanation.preference.review import blank_notes
from vqa_explanation.training.sft_data import target_for
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def review(tmp_path, monkeypatch):
    directory = tmp_path / "review"
    directory.mkdir()
    records = []
    for index in range(3):
        image_path = f"images/{index}.jpg"
        image = tmp_path / image_path
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), color=(index * 30, 0, 0)).save(image)
        records.append({
            "question_id": f"q{index}", "image_id": f"i{index}", "image_path": image_path,
            "question": "Con gì?", "answer": "mèo", "explanations": ["Có lông."],
            "split": "train" if index < 2 else "test",
        })
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, records)
    inputs = records[:2]
    protocol = {"generation": "greedy"}
    manifest = {
        "inputs": {
            "experiment_id": "M1_SFT", "adapter": "checkpoint-3683",
            "model_id": "fixture", "model_revision": "revision", "split": "train", "seed": 42,
            "adapter_files": {"adapter_model.safetensors": "a" * 64},
            "context": {"purpose": "preference_candidates_pending_review",
                        "annotations_sha256": sha256_file(annotations)},
            "image_hashes": {row["question_id"]: sha256_file(tmp_path / row["image_path"])
                             for row in inputs},
        }, "protocol": protocol,
    }
    predictions = [{
        **{key: row[key] for key in ("question_id", "image_id", "image_path", "question", "split")},
        **{key: manifest["inputs"][key] for key in (
            "model_id", "model_revision", "experiment_id", "adapter", "seed")},
        **protocol, "reference_answer": row["answer"],
        "reference_explanations": row["explanations"],
        "raw_output": "Đáp án: mèo\nGiải thích: Có cánh.", "prediction_answer": "mèo",
        "prediction_explanation": "Có cánh.", "format_valid": True, "latency_seconds": 0.1,
        "image_sha256": manifest["inputs"]["image_hashes"][row["question_id"]],
    } for row in inputs]
    write_jsonl(directory / "inputs.jsonl", inputs)
    write_jsonl(directory / "candidates.jsonl", predictions)
    write_json(directory / "candidates.jsonl.manifest.json", manifest)
    payloads = {path.name: path.read_bytes() for path in directory.iterdir()}
    payloads.update({row["image_path"]: (tmp_path / row["image_path"]).read_bytes()
                     for row in inputs})
    bundle = tmp_path / "review.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr("checksums.json", json.dumps({
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
        }))
    digest = sha256_file(bundle)
    baseline = blank_notes(inputs, {row["question_id"]: row for row in predictions}, digest)
    notes = copy.deepcopy(baseline)
    write_jsonl(directory / "review_notes.jsonl", baseline)
    pilot_path = tmp_path / "pilot.json"
    write_json(pilot_path, {
        "schema": "preference_review_pilot_v1", "count": 2, "question_ids": ["q0", "q1"],
        **{key: sha256_file(directory / name) for key, name in (
            ("source_inputs_sha256", "inputs.jsonl"),
            ("source_candidates_sha256", "candidates.jsonl"),
            ("source_notes_sha256", "review_notes.jsonl"),
        )},
    })
    notes_path = tmp_path / "notes.jsonl"
    write_jsonl(notes_path, notes)

    def no_model(*args, **kwargs):
        raise AssertionError("Review handoff must not load or train a model")

    monkeypatch.setattr("vqa_explanation.inference.export.Qwen25VLRunner", no_model)
    return SimpleNamespace(root=tmp_path, directory=directory, notes=notes, notes_path=notes_path,
                           pilot=pilot_path, bundle=bundle, digest=digest, annotations=annotations)


def resolve_review(review, *, second="rejected"):
    review.notes[0].update(
        review_status="accepted", reviewer="test-fixture-not-a-person",
        reviewed_at="2026-10-12T12:00:00+07:00", answer_assessment="correct",
        explanation_support=0, canonical_answer_verified=True, unsupported_span="cánh",
        image_evidence="Có lông, không có cánh.", proposed_chosen_explanation="Có lông.",
        proposed_rejected_explanation="Có cánh.", error_type="attribute",
        rejected_origin_kind="model", chosen_origin_kind="synthetic",
        chosen_origin_source="fixture-recipe:q0", chosen_recipe_sha256="c" * 64,
    )
    if second != "pending":
        review.notes[1].update(
            review_status=second, reviewer="test-fixture-not-a-person",
            reviewed_at="2026-10-12T12:00:00+07:00", exclusion_reason="Không đủ đối lập.",
        )
    write_jsonl(review.notes_path, review.notes)


def run(review, name="handoff", **kwargs):
    return build_handoff(review.directory, review.notes_path, review.pilot,
                         review.root / name, **kwargs)


def freeze_args(review, expected_pairs=1):
    return dict(freeze=True, expected_pairs=expected_pairs, bundle=review.bundle,
                expected_bundle_sha256=review.digest, annotations=review.annotations,
                image_root=review.root)


def test_pending_notes_remain_pending_and_cannot_be_frozen(review):
    before = review.notes_path.read_bytes()
    report = run(review, **freeze_args(review))
    assert report["status"] == "needs_pilot_review"
    assert len(report["freeze_blockers"]) == 3
    assert report["pending_pilot_ids"] == ["q0", "q1"]
    assert report["training_gate_open"] is False and report["gpu_training_started"] is False
    assert report["frozen_bundle"] is None
    assert not (review.root / "handoff/frozen").exists()
    assert review.notes_path.read_bytes() == before
    assert list(read_jsonl(review.root / "handoff/notes_snapshot.jsonl")) == review.notes


def test_complete_pair_draft_is_not_counted_as_accepted(review):
    resolve_review(review)
    review.notes[0].update(review_status="pending", reviewer=None, reviewed_at=None)
    write_jsonl(review.notes_path, review.notes)
    report = run(review)
    assert report["pair_draft_ids"] == ["q0"]
    assert report["review_counts"].get("accepted", 0) == 0
    assert report["pending_pilot_ids"] == ["q0"]
    assert report["pilot"]["resolved"] == 1


def test_complete_review_is_not_frozen_without_explicit_request(review):
    resolve_review(review)
    report = run(review)
    assert report["status"] == "ready_to_lock_pair_budget"
    assert report["frozen_bundle"] is None
    assert not (review.root / "handoff/export").exists()


def test_real_export_and_freeze_keep_same_chosen_and_origin(review):
    resolve_review(review)
    before = review.notes_path.read_bytes()
    report = run(review, **freeze_args(review))
    assert report["status"] == "frozen_metadata" and report["frozen_pairs"] == 1
    assert report["training_gate_open"] is False
    paired = list(read_jsonl(review.root / "handoff/frozen/accepted.jsonl"))
    chosen = list(read_jsonl(review.root / "handoff/frozen/chosen_only.jsonl"))
    assert paired[0]["chosen"] == target_for(chosen[0])
    assert paired[0]["provenance"]["chosen"]["kind"] == "synthetic"
    assert paired[0]["provenance"]["rejected"]["kind"] == "model"
    assert report["files"]["frozen/accepted.jsonl"] == sha256_file(
        review.root / "handoff/frozen/accepted.jsonl")
    assert review.notes_path.read_bytes() == before


def test_unused_pending_pool_does_not_prevent_freeze_of_locked_subset(review):
    resolve_review(review, second="pending")
    pilot = json.loads(review.pilot.read_text(encoding="utf-8"))
    pilot.update(count=1, question_ids=["q0"])
    write_json(review.pilot, pilot)
    report = run(review, **freeze_args(review))
    assert report["status"] == "frozen_metadata"
    assert report["review_counts"]["pending"] == 1
    assert report["frozen_pairs"] == 1


@pytest.mark.parametrize("expected", [2, 1000])
def test_budget_mismatch_does_not_export_subset_or_change_n(review, expected):
    resolve_review(review)
    report = run(review, **freeze_args(review, expected))
    assert report["frozen_bundle"] is None
    assert "expected_pairs" in report["freeze_blockers"][0]
    assert not (review.root / "handoff/export").exists()


def test_fake_acceptance_without_reviewer_cannot_export(review):
    resolve_review(review)
    review.notes[0]["reviewer"] = None
    write_jsonl(review.notes_path, review.notes)
    report = run(review, **freeze_args(review))
    assert report["status"] == "invalid"
    assert not (review.root / "handoff/frozen").exists()


def test_changed_bundle_fails_without_publishing_a_partial_handoff(review):
    resolve_review(review)
    review.bundle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        run(review, **freeze_args(review))
    assert not (review.root / "handoff").exists()
    assert not list(review.root.glob(".review-handoff-*"))


def test_test_split_cannot_enter_frozen_training(review):
    resolve_review(review)
    records = list(read_jsonl(review.annotations))
    records[0]["split"] = "test"
    write_jsonl(review.annotations, records)
    with pytest.raises(ValueError, match="canonical annotation"):
        run(review, **freeze_args(review))
    assert not (review.root / "handoff").exists()


def test_historical_handoff_is_preserved(review):
    run(review)
    report_path = review.root / "handoff/handoff.json"
    before = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        run(review)
    assert report_path.read_bytes() == before


@pytest.mark.parametrize("field", ["inputs.jsonl", "candidates.jsonl", "review_notes.jsonl"])
def test_changed_pilot_anchor_fails_before_publishing(review, field):
    (review.directory / field).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        run(review)
    assert not (review.root / "handoff").exists()


@pytest.mark.parametrize("ids,count", [(["q0", "q0"], 2), (["q0"], 2), ([], 0), ([[]], 1)])
def test_invalid_fixed_pilot_is_rejected(review, ids, count):
    pilot = json.loads(review.pilot.read_text(encoding="utf-8"))
    pilot.update(question_ids=ids, count=count)
    write_json(review.pilot, pilot)
    with pytest.raises(ValueError, match="unique question IDs"):
        run(review)
    assert not (review.root / "handoff").exists()


def test_source_change_during_export_is_not_published(review, monkeypatch):
    resolve_review(review)
    from vqa_explanation.preference import handoff
    original = handoff.freeze_pairs

    def changed(*args):
        result = original(*args)
        review.notes_path.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(handoff, "freeze_pairs", changed)
    with pytest.raises(ValueError, match="inputs changed"):
        run(review, **freeze_args(review))
    assert not (review.root / "handoff").exists()


@pytest.mark.parametrize("kwargs", [
    {"freeze": True}, {"expected_pairs": 1}, {"target_pairs": 0}, {"target_pairs": True},
])
def test_invalid_mode_or_budget_is_rejected(review, kwargs):
    with pytest.raises(ValueError):
        run(review, **kwargs)
    assert not (review.root / "handoff").exists()


def test_cli_reports_blocked_freeze_without_loading_gpu(review, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "finalize_preference_review.py", "--review-dir", str(review.directory),
        "--notes", str(review.notes_path), "--pilot", str(review.pilot),
        "--output-dir", str(review.root / "cli-handoff"), "--freeze", "--expected-pairs", "1",
        "--bundle", str(review.bundle), "--expected-bundle-sha256", review.digest,
        "--annotations", str(review.annotations), "--image-root", str(review.root),
    ])
    assert finalize_preference_review.main() == 2
    assert json.loads(capsys.readouterr().out)["training_gate_open"] is False


def test_cli_output_is_safe_on_windows_cp1258_console(review, monkeypatch):
    stream = io.BytesIO()
    console = io.TextIOWrapper(stream, encoding="cp1258")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "argv", [
        "finalize_preference_review.py", "--review-dir", str(review.directory),
        "--notes", str(review.notes_path), "--pilot", str(review.pilot),
        "--output-dir", str(review.root / "windows-handoff"),
    ])
    assert finalize_preference_review.main() == 0
    console.flush()
    report = json.loads(stream.getvalue().decode("cp1258"))
    assert report["status"] == "needs_pilot_review"
    assert "Pilot cố định" in report["freeze_blockers"][0]
