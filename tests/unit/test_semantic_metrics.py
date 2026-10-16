import hashlib
import json

import pytest

from vqa_explanation.evaluation.semantic_metrics import (
    DEFAULT_BERTSCORE_REVISION,
    audit_bertscore_lengths,
    compare_bertscore_reports,
    prepare_bertscore_inputs,
    resolve_bertscore_model,
    score_bertscore,
)


def prediction(question_id="q1", image_id="i1", explanation="Có một con mèo."):
    return {
        "question_id": question_id,
        "image_id": image_id,
        "prediction_explanation": explanation,
        "reference_explanations": ["Trong ảnh có một con mèo.", "Có mèo trong ảnh."],
    }


def versions(name):
    return {
        "bert-score": "0.3.13",
        "transformers": "4.57.6",
        "torch": "2.8.0",
    }[name]


def test_prepare_bertscore_inputs_keeps_empty_predictions_and_all_references():
    rows = [prediction(explanation=""), prediction("q2", "i2")]

    ids, image_ids, candidates, references = prepare_bertscore_inputs(rows)

    assert ids == ["q1", "q2"]
    assert image_ids == ["i1", "i2"]
    assert candidates == ["", "Có một con mèo."]
    assert references[0] == ["Trong ảnh có một con mèo.", "Có mèo trong ảnh."]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows.append(prediction()), "Duplicate question_id"),
        (lambda rows: rows[0].update(question_id=""), "question_id"),
        (lambda rows: rows[0].update(image_id=None), "image_id"),
        (lambda rows: rows[0].update(prediction_explanation=None), "prediction_explanation"),
        (lambda rows: rows[0].update(reference_explanations=[]), "reference_explanations"),
    ],
)
def test_prepare_bertscore_inputs_rejects_incomplete_rows(change, message):
    rows = [prediction()]
    change(rows)
    with pytest.raises(ValueError, match=message):
        prepare_bertscore_inputs(rows)


def test_resolve_bertscore_model_requires_commit_and_model_files(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    resolved = resolve_bertscore_model(
        "google-bert/bert-base-multilingual-cased",
        DEFAULT_BERTSCORE_REVISION,
        cache_dir=tmp_path / "cache",
        downloader=downloader,
    )

    assert resolved == snapshot
    assert calls[0]["revision"] == DEFAULT_BERTSCORE_REVISION
    assert "config.json" in calls[0]["allow_patterns"]
    assert "pytorch_model.bin" not in calls[0]["allow_patterns"]
    with pytest.raises(ValueError, match="commit SHA"):
        resolve_bertscore_model("model", "main", downloader=downloader)


class LengthTokenizer:
    model_max_length = 8

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True
        assert truncation is False
        return [0] * (len(text.split()) + 2)


def test_length_audit_counts_special_tokens_all_references_and_empty_answers(tmp_path):
    rows = [prediction(explanation="  ")]
    rows[0]["reference_explanations"] = ["one", "one two three four five six"]
    result = audit_bertscore_lengths(rows, model_path=tmp_path, tokenizer=LengthTokenizer())
    assert result == {
        "n": 1, "reference_count": 2, "model_max_length": 8,
        "truncated_texts": 0, "candidate_max_tokens": 2, "reference_max_tokens": 8,
    }


@pytest.mark.parametrize("field", ["prediction_explanation", "reference_explanations"])
def test_length_audit_rejects_overflow_in_candidates_or_any_reference(tmp_path, field):
    rows = [prediction(explanation="short")]
    rows[0]["reference_explanations"] = ["short"]
    long_text = "one two three four five six seven"
    rows[0][field] = long_text if field == "prediction_explanation" else ["short", long_text]
    with pytest.raises(ValueError, match="would truncate.*q1: 9 > 8"):
        audit_bertscore_lengths(rows, model_path=tmp_path, tokenizer=LengthTokenizer())


def test_score_bertscore_records_every_item_and_pins_protocol(tmp_path):
    rows = [prediction(), prediction("q2", "i2", "Một con chó.")]
    calls = []

    def fake_score(candidates, references, **kwargs):
        calls.append((candidates, references, kwargs))
        return (
            ([0.8, 0.6], [0.7, 0.5], [0.75, 0.55]),
            f"{kwargs['model_type']}_L9_no-idf_version=0.3.12(hug_trans=4.57.6)",
        )

    first = score_bertscore(
        rows,
        model_path=tmp_path,
        scorer=fake_score,
        version_reader=versions,
        device="cpu",
        batch_size=2,
    )
    second_path = tmp_path / "other-cache-path"
    second_path.mkdir()
    second = score_bertscore(
        rows,
        model_path=second_path,
        scorer=fake_score,
        version_reader=versions,
        device="cpu",
        batch_size=2,
    )

    assert first["n"] == 2
    assert first["mean_f1"] == pytest.approx(0.65)
    assert [item["question_id"] for item in first["items"]] == ["q1", "q2"]
    assert first["items"][0]["f1"] == 0.75
    assert first["protocol"]["model_revision"] == DEFAULT_BERTSCORE_REVISION
    assert first["protocol"]["bert_score_version"] == "0.3.13"
    assert first["protocol"]["backend_reported_bert_score_version"] == "0.3.12"
    assert first["backend_settings"].startswith(
        "google-bert/bert-base-multilingual-cased_L9_no-idf"
    )
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert len(first["protocol_sha256"]) == 64
    assert hashlib.sha256(
        json.dumps(first["protocol"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest() == first["protocol_sha256"]
    assert calls[0][2]["model_type"] == str(tmp_path)
    assert calls[0][2]["num_layers"] == 9
    assert calls[0][2]["use_fast_tokenizer"] is False
    assert calls[0][1][0] == rows[0]["reference_explanations"]
    assert "does not verify visual grounding" in first["limitations"]


def test_score_bertscore_rejects_unpinned_runtime(tmp_path):
    def wrong_version(name):
        return "0.3.12" if name == "bert-score" else versions(name)

    with pytest.raises(RuntimeError, match="Expected bert-score 0.3.13"):
        score_bertscore(
            [prediction()],
            model_path=tmp_path,
            scorer=lambda *args, **kwargs: None,
            version_reader=wrong_version,
        )


def test_compare_bertscore_reports_binds_predictions_protocol_ids_and_images(tmp_path):
    rows = [prediction(), prediction("q2", "i2")]

    def scorer(values):
        def fake_score(candidates, references, **kwargs):
            return (
                (values, values, values),
                f"{kwargs['model_type']}_L9_no-idf_version=0.3.12(hug_trans=4.57.6)",
            )

        return fake_score

    left_score = score_bertscore(
        rows,
        model_path=tmp_path,
        scorer=scorer([0.5, 0.7]),
        version_reader=versions,
    )
    right_score = score_bertscore(
        rows,
        model_path=tmp_path,
        scorer=scorer([0.6, 0.9]),
        version_reader=versions,
    )
    left = {
        "n": 2,
        "split": "validation",
        "predictions_sha256": "a" * 64,
        "explanation_bertscore": left_score,
    }
    right = {
        "n": 2,
        "split": "validation",
        "predictions_sha256": "b" * 64,
        "explanation_bertscore": right_score,
    }

    report, differences, image_ids = compare_bertscore_reports(
        left,
        right,
        left_predictions_sha256="a" * 64,
        right_predictions_sha256="b" * 64,
        expected_split="validation",
        expected_question_images={"q1": "i1", "q2": "i2"},
    )

    assert report["comparable"]
    assert report["delta_right_minus_left"]["f1"] == pytest.approx(0.15)
    assert differences == pytest.approx([0.1, 0.2])
    assert image_ids == ["i1", "i2"]
    left["predictions_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different predictions"):
        compare_bertscore_reports(
            left,
            right,
            left_predictions_sha256="a" * 64,
            right_predictions_sha256="b" * 64,
            expected_split="validation",
            expected_question_images={"q1": "i1", "q2": "i2"},
        )
