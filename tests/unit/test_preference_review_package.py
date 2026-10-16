import hashlib
import json
import sys
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import package_preference_review as package_script
from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.inference.export import select_records
from vqa_explanation.preference.bundle import package_preference_review, verify_bundle
from vqa_explanation.preference.selection import freeze_selection
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def review_package(tmp_path, monkeypatch):
    rows = []
    images = tmp_path / "images"
    images.mkdir()
    for index, split in enumerate(("train", "train", "train", "validation", "test")):
        image_path = f"images/{index}.jpg"
        Image.new("RGB", (4, 4), color=(index * 20, 0, 0)).save(tmp_path / image_path)
        rows.append(dict(question_id=f"q{index}", image_id=f"i{index}",
                         image_path=image_path, split=split, question="Con gì?",
                         answer="mèo", explanations=["Có mèo."]))
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, rows)
    pool_dir = tmp_path / "pool"
    freeze_selection(annotations, tmp_path, pool_dir, count=3)
    questions = list(read_jsonl(pool_dir / "questions.jsonl"))
    by_id = {row["question_id"]: row for row in rows}
    records = [by_id[row["source_question_id"]] for row in questions[:2]]
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    write_jsonl(candidate_dir / "inputs.jsonl", records)
    files = {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
    adapter = {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()}
    m1_selection = tmp_path / "selection.json"
    write_json(m1_selection, dict(status="selected", selection_split="validation",
                                 annotations_sha256=sha256_file(annotations), adapter=adapter,
                                 model_id="model", model_revision="revision"))
    args = dict(
        candidate_dir=candidate_dir, pool_dir=pool_dir, m1_selection=m1_selection,
        image_root=tmp_path, output=tmp_path / "review.zip", expected_count=2,
        annotations_sha256=sha256_file(annotations),
        pool_questions_sha256=sha256_file(pool_dir / "questions.jsonl"),
        pool_manifest_sha256=sha256_file(pool_dir / "manifest.json"),
        m1_selection_sha256=sha256_file(m1_selection),
    )
    inputs = dict(
        model_id="model", model_revision="revision", experiment_id="M1_SFT",
        adapter="/historical/checkpoint-3683", adapter_files=files, split="train", limit=0,
        seed=42, annotations_sha256=sha256_file(candidate_dir / "inputs.jsonl"),
        selected_sha256=hashlib.sha256(json.dumps(
            select_records(records, "train", 0, 42), ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest(),
        image_hashes={row["question_id"]: sha256_file(tmp_path / row["image_path"])
                      for row in records},
        context=dict(purpose="preference_candidates_pending_review", count=2,
                     selection_order="frozen_prefix", annotations_sha256=args["annotations_sha256"],
                     selection_sha256=args["pool_questions_sha256"],
                     selection_manifest_sha256=args["pool_manifest_sha256"],
                     m1_selection_sha256=args["m1_selection_sha256"], m1_adapter=adapter),
    )
    protocol = {"generation": "greedy"}
    write_json(candidate_dir / "candidates.jsonl.manifest.json",
               dict(schema_version=1, inputs=inputs, protocol=protocol))
    predictions = [dict(
        **{key: row[key] for key in ("question_id", "image_id", "image_path", "question", "split")},
        **{key: inputs[key] for key in ("model_id", "model_revision", "experiment_id",
                                       "adapter", "seed")},
        **protocol, image_sha256=inputs["image_hashes"][row["question_id"]],
        reference_answer=row["answer"], reference_explanations=row["explanations"],
        raw_output="Đáp án: mèo\nGiải thích: Có mèo.", prediction_answer="mèo",
        prediction_explanation="Có mèo.", format_valid=True, latency_seconds=0.1,
    ) for row in reversed(records)]
    write_jsonl(candidate_dir / "candidates.jsonl", predictions)
    refresh_summary(args)
    (candidate_dir / "console.log").write_text("Do not bundle this log", encoding="utf-8")
    (candidate_dir / ".env").write_text("SECRET=not-a-real-token", encoding="utf-8")

    def no_model(*args, **kwargs):
        raise AssertionError("Review packaging must not load a model")

    monkeypatch.setattr("vqa_explanation.inference.export.Qwen25VLRunner", no_model)
    return SimpleNamespace(args=args, records=records, questions=questions)


def refresh_summary(args):
    directory = args["candidate_dir"]
    write_json(directory / "summary.json", dict(
        status="generated_pending_review", n=2, format_valid=2, reference_answer_em=1.0,
        **{field: sha256_file(directory / name) for field, name in (
            ("inputs_sha256", "inputs.jsonl"), ("predictions_sha256", "candidates.jsonl"),
            ("manifest_sha256", "candidates.jsonl.manifest.json"),
        )},
    ))


def edit_manifest(args, mutate):
    path = args["candidate_dir"] / "candidates.jsonl.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    write_json(path, manifest)
    refresh_summary(args)


def retarget_image(args, image_path, index=0):
    """Change all recorded paths consistently so archive safety is the failing gate."""
    questions_path = args["pool_dir"] / "questions.jsonl"
    questions = list(read_jsonl(questions_path))
    question_id = questions[index]["source_question_id"]
    questions[index]["image_path"] = image_path
    write_jsonl(questions_path, questions)
    args["pool_questions_sha256"] = sha256_file(questions_path)
    pool_manifest_path = args["pool_dir"] / "manifest.json"
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    pool_manifest["files"]["questions.jsonl"] = args["pool_questions_sha256"]
    write_json(pool_manifest_path, pool_manifest)
    args["pool_manifest_sha256"] = sha256_file(pool_manifest_path)
    inputs_path = args["candidate_dir"] / "inputs.jsonl"
    records = list(read_jsonl(inputs_path))
    next(row for row in records if row["question_id"] == question_id)["image_path"] = image_path
    write_jsonl(inputs_path, records)
    predictions_path = args["candidate_dir"] / "candidates.jsonl"
    predictions = list(read_jsonl(predictions_path))
    next(row for row in predictions if row["question_id"] == question_id)["image_path"] = image_path
    write_jsonl(predictions_path, predictions)

    def update(manifest):
        manifest["inputs"]["annotations_sha256"] = sha256_file(inputs_path)
        manifest["inputs"]["selected_sha256"] = hashlib.sha256(json.dumps(
            select_records(records, "train", 0, 42), ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        manifest["inputs"]["context"].update(
            selection_sha256=args["pool_questions_sha256"],
            selection_manifest_sha256=args["pool_manifest_sha256"],
        )

    edit_manifest(args, update)


def test_package_has_exact_pending_review_files_images_and_valid_checksums(review_package):
    a = review_package.args
    result = package_preference_review(**a)
    data, inputs, predictions = verify_bundle(a["output"], result["sha256"])
    expected_names = {
        "inputs.jsonl", "candidates.jsonl", "candidates.jsonl.manifest.json", "summary.json",
        "pool/questions.jsonl", "pool/manifest.json", "m1_selection.json", "checksums.json",
        *[row["image_path"] for row in review_package.records],
    }
    assert set(data) == expected_names
    assert inputs == review_package.records and len(predictions) == 2
    assert result == dict(path=str(a["output"].resolve()), sha256=sha256_file(a["output"]),
                          count=2, status="generated_pending_review")
    assert data["m1_selection.json"] == a["m1_selection"].read_bytes()
    assert not any("chosen" in row or "rejected" in row for row in predictions.values())


def test_exact_existing_package_is_idempotent_without_rewriting(review_package):
    a = review_package.args
    first = package_preference_review(**a)
    before = a["output"].read_bytes(), a["output"].stat().st_mtime_ns
    assert package_preference_review(**a) == first
    assert (a["output"].read_bytes(), a["output"].stat().st_mtime_ns) == before


def test_existing_different_package_is_not_overwritten(review_package):
    a = review_package.args
    with zipfile.ZipFile(a["output"], "x") as archive:
        archive.writestr("different.txt", "preserve me")
    before = a["output"].read_bytes()
    with pytest.raises(FileExistsError, match="Existing review bundle differs"):
        package_preference_review(**a)
    assert a["output"].read_bytes() == before


def test_existing_non_zip_file_is_preserved(review_package):
    a = review_package.args
    a["output"].write_bytes(b"preserve existing bytes")
    with pytest.raises(zipfile.BadZipFile):
        package_preference_review(**a)
    assert a["output"].read_bytes() == b"preserve existing bytes"


def test_same_member_names_with_changed_bytes_are_not_overwritten(review_package):
    a = review_package.args
    package_preference_review(**a)
    with zipfile.ZipFile(a["output"]) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["summary.json"] += b" "
    a["output"].unlink()
    with zipfile.ZipFile(a["output"], "x") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    before = a["output"].read_bytes()
    with pytest.raises(FileExistsError, match="Existing review bundle differs"):
        package_preference_review(**a)
    assert a["output"].read_bytes() == before


@pytest.mark.parametrize("target", ["questions", "pool_manifest", "selection", "image"])
def test_changed_frozen_source_is_rejected(review_package, target):
    a = review_package.args
    paths = {"questions": a["pool_dir"] / "questions.jsonl",
             "pool_manifest": a["pool_dir"] / "manifest.json", "selection": a["m1_selection"],
             "image": a["image_root"] / review_package.records[0]["image_path"]}
    paths[target].write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        package_preference_review(**a)
    assert not a["output"].exists()


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("n", 1), ("n", True), ("inputs_sha256", "c" * 64),
    ("predictions_sha256", "c" * 64), ("manifest_sha256", "c" * 64),
])
def test_incomplete_or_tampered_summary_is_rejected(review_package, field, value):
    a = review_package.args
    path = a["candidate_dir"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[field] = value
    write_json(path, summary)
    with pytest.raises(ValueError, match="Summary"):
        package_preference_review(**a)
    assert not a["output"].exists()


@pytest.mark.parametrize("count", [1, 3, 0, True])
def test_wrong_requested_count_is_rejected(review_package, count):
    a = {**review_package.args, "expected_count": count}
    with pytest.raises(ValueError, match="count"):
        package_preference_review(**a)
    assert not a["output"].exists()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "reference", "parsed"])
def test_invalid_candidate_coverage_or_content_is_rejected_even_with_new_summary(
    review_package, fault,
):
    a = review_package.args
    path = a["candidate_dir"] / "candidates.jsonl"
    rows = list(read_jsonl(path))
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    else:
        rows[0][{"unknown": "question_id", "reference": "reference_answer",
                 "parsed": "prediction_explanation"}[fault]] = "changed"
    write_jsonl(path, rows)
    refresh_summary(a)
    with pytest.raises(ValueError, match="Incomplete|Saved prediction"):
        package_preference_review(**a)
    assert not a["output"].exists()


def test_missing_candidate_file_is_rejected_before_creating_zip(review_package):
    a = review_package.args
    (a["candidate_dir"] / "candidates.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        package_preference_review(**a)
    assert not a["output"].exists()


@pytest.mark.parametrize("fault", ["train_split", "reference", "prefix_order", "duplicate"])
def test_inputs_must_match_exact_frozen_train_prefix(review_package, fault):
    a = review_package.args
    path = a["candidate_dir"] / "inputs.jsonl"
    rows = list(read_jsonl(path))
    if fault == "train_split":
        rows[0]["split"] = "test"
    elif fault == "reference":
        rows[0]["answer"] = "different"
    elif fault == "prefix_order":
        rows.reverse()
    else:
        rows[1] = rows[0]
    write_jsonl(path, rows)
    refresh_summary(a)
    with pytest.raises(ValueError, match="exact frozen train prefix"):
        package_preference_review(**a)


@pytest.mark.parametrize("field", ["annotations_sha256", "selection_sha256",
                                   "selection_manifest_sha256", "m1_selection_sha256"])
def test_wrong_manifest_anchor_is_rejected_with_matching_summary(review_package, field):
    a = review_package.args
    edit_manifest(a, lambda m: m["inputs"]["context"].update({field: "c" * 64}))
    with pytest.raises(ValueError, match="frozen review context"):
        package_preference_review(**a)


@pytest.mark.parametrize("target", ["files", "aggregate", "model", "canonical_annotation"])
def test_m1_parent_must_match_frozen_selection(review_package, target):
    a = review_package.args

    def mutate(manifest):
        if target == "files":
            manifest["inputs"]["adapter_files"]["adapter_config.json"] = "c" * 64
        elif target == "aggregate":
            manifest["inputs"]["context"]["m1_adapter"]["sha256"] = "c" * 64
        elif target == "model":
            manifest["inputs"]["model_id"] = "other"
        else:
            a["annotations_sha256"] = "c" * 64

    edit_manifest(a, mutate)
    with pytest.raises(ValueError, match="frozen M1 selection|pool manifest"):
        package_preference_review(**a)


def test_another_same_count_pool_cannot_replace_the_expected_pool(review_package):
    a = review_package.args
    questions_path = a["pool_dir"] / "questions.jsonl"
    questions = list(read_jsonl(questions_path))
    questions.reverse()
    write_jsonl(questions_path, questions)
    manifest_path = a["pool_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["questions.jsonl"] = sha256_file(questions_path)
    manifest["calibration_question_ids"] = [row["source_question_id"] for row in questions]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Frozen anchor hash changed"):
        package_preference_review(**a)


@pytest.mark.parametrize("path", ["../outside.jpg", "/outside.jpg", "C:/outside.jpg",
                                  "images\\outside.jpg", "images//outside.jpg",
                                  "images/./outside.jpg"])
def test_unsafe_image_paths_are_rejected_even_in_an_anchored_pool(review_package, path):
    a = review_package.args
    questions_path = a["pool_dir"] / "questions.jsonl"
    questions = list(read_jsonl(questions_path))
    questions[0]["image_path"] = path
    write_jsonl(questions_path, questions)
    a["pool_questions_sha256"] = sha256_file(questions_path)
    manifest_path = a["pool_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["questions.jsonl"] = a["pool_questions_sha256"]
    write_json(manifest_path, manifest)
    a["pool_manifest_sha256"] = sha256_file(manifest_path)
    with pytest.raises(ValueError, match="Unsafe image path"):
        package_preference_review(**a)
    assert not a["output"].exists()


@pytest.mark.parametrize("path", ["inputs.jsonl", "summary.json", "pool",
                                  "pool/questions.jsonl/child.jpg", "CHECKSUMS.JSON",
                                  "index.html", "review_notes.jsonl"])
def test_metadata_and_renderer_path_collisions_are_rejected(review_package, path):
    a = review_package.args
    retarget_image(a, path)
    with pytest.raises(ValueError, match="collides with a bundle member"):
        package_preference_review(**a)
    assert not a["output"].exists()


def test_image_names_differing_only_by_case_are_rejected(review_package):
    a = review_package.args
    retarget_image(a, review_package.records[0]["image_path"].upper(), index=1)
    with pytest.raises(ValueError, match="collides with a bundle member"):
        package_preference_review(**a)
    assert not a["output"].exists()


def test_cli_requires_anchors_and_reports_package_hash_count_status(review_package, monkeypatch,
                                                                   capsys):
    a = review_package.args
    argv = ["package_preference_review.py"]
    for name, value in a.items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    package_script.main()
    report = json.loads(capsys.readouterr().out)
    assert report["sha256"] == sha256_file(a["output"])
    assert report["count"] == 2 and report["status"] == "generated_pending_review"
    monkeypatch.setattr(sys, "argv", ["package_preference_review.py"])
    with pytest.raises(SystemExit):
        package_script.main()
