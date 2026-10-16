import hashlib
import json

import pytest
from PIL import Image

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation.review import (
    prepare_validation_review,
    render_review,
    select_review_records,
)
from vqa_explanation.inference.export import select_records
from vqa_explanation.utils.run_provenance import sha256_file


def reference(qid, image_id):
    return dict(question_id=qid, image_id=image_id, image_path=f"images/{image_id}.jpg",
                question="Con gì?", answer="mèo", explanations=["Có mèo."], split="validation")


def test_selection_is_order_independent_and_one_question_per_image():
    records = [reference("q1", "1"), reference("q2", "1"), reference("q3", "2")]
    chosen = select_review_records(records, count=2)
    assert chosen == select_review_records(records[::-1], count=2)
    assert len({row["image_id"] for row in chosen}) == 2
    changed_answers = [{**row, "answer": "chó"} for row in records]
    assert [r["question_id"] for r in chosen] == [
        r["question_id"] for r in select_review_records(changed_answers, count=2)
    ]


@pytest.mark.parametrize("records,count", [
    ([reference("q1", "1")], 2),
    ([reference("q1", "1"), reference("q1", "2")], 1),
    ([{**reference("q1", "1"), "split": "train"}], 1),
    ([reference("q1", "1")], 0),
])
def test_selection_rejects_incomplete_or_wrong_split(records, count):
    with pytest.raises(ValueError):
        select_review_records(records, count=count)


@pytest.fixture
def inputs(tmp_path):
    records = [reference("q1", "1"), reference("q2", "2")]
    (tmp_path / "images").mkdir()
    for row in records:
        Image.new("RGB", (8, 8)).save(tmp_path / row["image_path"])
    annotations, predictions, manifest = [tmp_path / name for name in
                                          ("validation.jsonl", "m1.jsonl", "manifest.json")]
    write_jsonl(annotations, records)
    protocol = {"compute_dtype": "torch.float16", "max_new_tokens": 128}
    inputs = dict(model_id="model", model_revision="revision", experiment_id="M1_SFT",
                  adapter="runs/m1/checkpoint-3683", adapter_files={"weights": "a" * 64},
                  split="validation", limit=0, seed=42, annotations_sha256=sha256_file(annotations),
                  selected_sha256=hashlib.sha256(json.dumps(select_records(records, "validation",
                      0, 42), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                  image_hashes={r["question_id"]: sha256_file(tmp_path / r["image_path"])
                                for r in records})
    write_json(manifest, {"schema_version": 1, "inputs": inputs, "protocol": protocol})
    write_jsonl(predictions, [{
        **protocol, **{key: row[key] for key in ("question_id", "image_id", "image_path",
                                               "question", "split")},
        "image_sha256": inputs["image_hashes"][row["question_id"]],
        **{key: inputs[key] for key in ("model_id", "model_revision", "experiment_id",
                                       "adapter", "seed")},
        "reference_answer": row["answer"], "reference_explanations": row["explanations"],
        "raw_output": "Đáp án: mèo\nGiải thích: Có mèo.",
        "prediction_answer": "mèo", "prediction_explanation": "Có mèo.",
        "format_valid": True, "latency_seconds": 1.0,
    } for row in records])
    return dict(annotations=annotations, annotations_sha256=sha256_file(annotations),
                predictions=predictions, predictions_sha256=sha256_file(predictions),
                manifest=manifest, manifest_sha256=sha256_file(manifest), image_root=tmp_path,
                output_dir=tmp_path / "review", expected_count=2, count=2)


def test_bundle_has_verified_images_and_no_invented_labels(inputs):
    page = prepare_validation_review(**inputs)
    notes = list(read_jsonl(page.parent / "review_notes.jsonl"))
    assert len(notes) == 2
    assert all(r["review_status"] == "pending" and r["reviewer"] is None for r in notes)
    assert all(r["answer_assessment"] is None and r["explanation_support"] is None for r in notes)
    receipt = json.loads((page.parent / "manifest.json").read_text())
    assert receipt["status"] == "pending_human_review"
    assert len(receipt["selection"]["image_ids"]) == 2
    for name, digest in receipt["files"].items():
        assert sha256_file(page.parent / name) == digest
    with pytest.raises(FileExistsError):
        prepare_validation_review(**inputs)


@pytest.mark.parametrize("pin", ["annotations_sha256", "predictions_sha256", "manifest_sha256"])
def test_review_rejects_wrong_input_hashes(inputs, pin):
    inputs[pin] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_validation_review(**inputs)
    assert not inputs["output_dir"].exists()


def test_review_rejects_changed_image_before_writing(inputs):
    Image.new("RGB", (8, 8), "red").save(inputs["image_root"] / "images/1.jpg")
    with pytest.raises(ValueError, match="differs from inference"):
        prepare_validation_review(**inputs)
    assert not inputs["output_dir"].exists()


def test_review_requires_full_prediction_coverage(inputs):
    rows = list(read_jsonl(inputs["predictions"]))
    write_jsonl(inputs["predictions"], rows[:1])
    inputs["predictions_sha256"] = sha256_file(inputs["predictions"])
    with pytest.raises(ValueError, match="coverage"):
        prepare_validation_review(**inputs)


def test_review_rejects_changed_parsed_output(inputs):
    rows = list(read_jsonl(inputs["predictions"]))
    rows[0]["prediction_answer"] = "chó"
    write_jsonl(inputs["predictions"], rows)
    inputs["predictions_sha256"] = sha256_file(inputs["predictions"])
    with pytest.raises(ValueError, match="parsed output"):
        prepare_validation_review(**inputs)


def test_review_page_escapes_prediction_text(inputs):
    page = prepare_validation_review(**inputs)
    notes = list(read_jsonl(page.parent / "review_notes.jsonl"))
    notes[0]["prediction_explanation"] = "</script><script>alert(1)</script>"
    rendered = render_review(notes)
    assert "</script><script>alert(1)" not in rendered
    payload = rendered.split('id="data">')[1].split("</script>")[0]
    assert json.loads(payload) == notes
    assert "pending" in rendered and "m1_review_100.jsonl" in rendered
