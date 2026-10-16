import json

import pytest
from PIL import Image

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.evaluation.image_dependence import (
    prepare_images,
    run_image_check,
    score_image_check,
    verify_images,
)
from vqa_explanation.inference.export import export_predictions
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def diagnostic(tmp_path):
    rows = []
    for i in range(8):
        Image.new("RGB", (8, 8), (i * 25, 0, 0)).save(tmp_path / f"{i}.jpg")
        rows.append({"question_id": f"q{i}", "image_id": f"i{i}",
                     "image_path": f"{i}.jpg", "question": f"Ảnh {i}?", "answer": str(i),
                     "explanations": [f"Có vật {i}."],
                     "split": "train" if i == 0 else "test" if i == 7 else "validation"})
    annotations = tmp_path / "all.jsonl"
    write_jsonl(annotations, rows)
    bundle = tmp_path / "diagnostic"
    prepare_images(annotations, tmp_path, bundle, 6)
    return annotations, tmp_path, bundle


def test_permutation_is_validation_only_fixed_and_does_not_change_questions(diagnostic):
    annotations, root, bundle = diagnostic
    before = sha256_file(annotations)
    manifest, data = verify_images(bundle, annotations, root)
    original, permuted = data.values()
    assert manifest["count"] == 6
    for left, right in zip(original, permuted, strict=True):
        assert left["split"] == right["split"] == "validation"
        assert left["question_id"] == right["question_id"]
        assert left["image_id"] != right["image_id"]
        assert left["answer"] == right["answer"]
    other = root / "other"
    second = prepare_images(annotations, root, other, 6)
    assert second == manifest
    assert before == sha256_file(annotations)
    with pytest.raises(FileExistsError):
        prepare_images(annotations, root, bundle, 6)


def test_verifier_rejects_mapping_changes_even_with_updated_file_hash(diagnostic):
    annotations, root, bundle = diagnostic
    path = bundle / "permuted_images.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["image_id"] = "unknown"
    write_jsonl(path, rows)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"][path.name] = sha256_file(path)
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mapping"):
        verify_images(bundle, annotations, root)


def test_verifier_rejects_changed_image_bytes(diagnostic):
    annotations, root, bundle = diagnostic
    (root / "1.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image bytes"):
        verify_images(bundle, annotations, root)


def test_run_score_resume_and_identity_checks_without_gpu(diagnostic):
    annotations, root, bundle = diagnostic

    class Runner:
        def protocol_metadata(self, count):
            return {"generation": "fixture_greedy", "max_new_tokens": count}

        def generate(self, image, question, max_new_tokens):
            return f"Đáp án: {image.stem}\nGiải thích: Có vật."

    def exporter(args):
        export_predictions(args, runner_factory=lambda *a, **kw: Runner())

    output = root / "predictions"
    run_image_check(bundle, annotations, root, output, "M0_PRETRAINED", None,
                    exporter=exporter)
    result = score_image_check(bundle, annotations, root, output)
    assert result["correct_image_em"] == 1
    assert result["permuted_image_em_against_original_answer"] == 0
    assert result["answer_change_rate"] == 1
    assert result["em_drop_ci95"] == [1, 1]
    run_image_check(bundle, annotations, root, output, "M0_PRETRAINED", None,
                    exporter=lambda args: export_predictions(
                        args, runner_factory=lambda *a, **kw: pytest.fail("Unexpected model load")
                    ))
    path = output / "permuted_images.jsonl.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"]["versions"]["torch"] = "different"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="identities"):
        score_image_check(bundle, annotations, root, output)


def test_incomplete_exports_are_not_reported_as_results(diagnostic):
    annotations, root, bundle = diagnostic
    with pytest.raises(FileNotFoundError):
        score_image_check(bundle, annotations, root, root / "not_run")
