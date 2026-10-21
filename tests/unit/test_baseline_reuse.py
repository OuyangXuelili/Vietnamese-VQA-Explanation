import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vqa_explanation.inference.baseline_reuse import (
    BASELINE_FILES,
    prepare_baseline_reuse,
    write_baseline_reuse,
)
from vqa_explanation.inference.export import export_predictions, read_jsonl
from vqa_explanation.utils.run_provenance import sha256_file


class FakeRunner:
    def __init__(self, *args, **kwargs):
        pass

    def protocol_metadata(self, max_new_tokens):
        return {"min_pixels": 200704, "max_pixels": 200704, "max_new_tokens": max_new_tokens,
                "load_in_4bit": True, "prompt_sha256": "a" * 64,
                "generation_config": {"do_sample": False, "max_new_tokens": max_new_tokens},
                "compute_dtype": "torch.float16"}

    def generate(self, *args, **kwargs):
        return "Đáp án: mèo\nGiải thích: Có mèo."


@pytest.fixture
def baselines(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.jpg")
    records = [dict(question_id=f"q{index}", image_id="im1", image_path="image.jpg",
                    split="validation", question="Con gì?", answer="mèo",
                    explanations=["Có mèo."]) for index in range(3)]
    annotations = tmp_path / "validation.jsonl"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    adapter = tmp_path / "m1"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"same weights")
    source, output = tmp_path / "baseline", tmp_path / "lr_validation"
    arguments = {}
    for arm, experiment in (("m0", "M0_PRETRAINED"), ("m1", "M1_SFT")):
        args = SimpleNamespace(
            annotations=annotations, image_root=tmp_path, output=source / f"{arm}.jsonl",
            split="validation", limit=0, seed=42, model_id="model", revision="revision",
            max_new_tokens=128, min_pixels=200704, max_pixels=200704, load_in_4bit=True,
            adapter=adapter if arm == "m1" else None, experiment_id=experiment,
        )
        export_predictions(args, runner_factory=FakeRunner)
        args.output = output / f"{arm}.jsonl"
        arguments[arm] = args
    hashes = {name: sha256_file(source / name) for name in BASELINE_FILES}
    return source, hashes, arguments, output


def test_reuse_preserves_source_bytes_and_skips_model_loading(baselines):
    source, hashes, arguments, output = baselines
    plan = prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()
    receipt = write_baseline_reuse(plan)
    assert receipt["destination_sha256"] == hashes
    assert all(sha256_file(source / name) == digest for name, digest in hashes.items())
    assert all((output / name).read_bytes() == (source / name).read_bytes() for name in hashes)
    assert {value["predictions"] for value in receipt["arms"].values()} == {3}
    before = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    write_baseline_reuse(prepare_baseline_reuse(source, hashes, arguments))
    assert before == {path.name: path.stat().st_mtime_ns for path in output.iterdir()}


@pytest.mark.parametrize("fault", ["missing", "extra", "invalid", "checksum", "absent"])
def test_requires_exact_pinned_hashes_before_any_write(baselines, fault):
    source, hashes, arguments, output = baselines
    if fault == "missing":
        hashes.pop("m0.jsonl")
    elif fault == "extra":
        hashes["m2.jsonl"] = "a" * 64
    elif fault == "invalid":
        hashes["m0.jsonl"] = "bad"
    elif fault == "checksum":
        (source / "m0.jsonl").write_bytes(b"changed")
    else:
        (source / "m1.jsonl").unlink()
    with pytest.raises(ValueError):
        prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("seed", 43), ("revision", "other"),
                                       ("max_new_tokens", 64), ("load_in_4bit", False)])
def test_rejects_inference_identity_changes(baselines, field, value):
    source, hashes, arguments, output = baselines
    setattr(arguments["m0"], field, value)
    with pytest.raises(ValueError, match="identity|protocol"):
        prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["weights", "image", "annotations"])
def test_rejects_changed_data_or_m1(baselines, fault):
    source, hashes, arguments, output = baselines
    args = arguments["m1"]
    if fault == "weights":
        (args.adapter / "adapter_model.safetensors").write_bytes(b"changed")
    elif fault == "image":
        Image.new("RGB", (8, 8)).save(args.image_root / "image.jpg")
    else:
        rows = read_jsonl(args.annotations)
        rows[0]["question"] = "Changed?"
        args.annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="identity"):
        prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["incomplete", "newline", "duplicate", "parsed"])
def test_rejects_incomplete_or_corrupt_predictions_even_if_repinned(baselines, fault):
    source, hashes, arguments, output = baselines
    path = source / "m0.jsonl"
    rows = read_jsonl(path)
    if fault == "incomplete":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "parsed":
        rows[0]["prediction_answer"] = "wrong"
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload.rstrip() if fault == "newline" else payload, encoding="utf-8")
    hashes[path.name] = sha256_file(path)
    with pytest.raises(ValueError):
        prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()


def test_rejects_different_baseline_prompt_protocols(baselines):
    source, hashes, arguments, output = baselines
    path = source / "m1.jsonl.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["protocol"]["prompt_sha256"] = "b" * 64
    path.write_text(json.dumps(manifest))
    hashes[path.name] = sha256_file(path)
    # Stored rows must also match their manifest; changing both still cannot mix prompts.
    predictions = source / "m1.jsonl"
    rows = read_jsonl(predictions)
    for row in rows:
        row["prompt_sha256"] = "b" * 64
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hashes[predictions.name] = sha256_file(predictions)
    with pytest.raises(ValueError, match="protocols differ"):
        prepare_baseline_reuse(source, hashes, arguments)
    assert not output.exists()


def test_adapter_storage_move_requires_identical_weights(baselines, tmp_path):
    source, hashes, arguments, output = baselines
    adapter = tmp_path / "moved_m1"
    adapter.mkdir()
    for path in arguments["m1"].adapter.iterdir():
        (adapter / path.name).write_bytes(path.read_bytes())
    arguments["m1"].adapter = adapter
    receipt = write_baseline_reuse(prepare_baseline_reuse(source, hashes, arguments))
    assert receipt["arms"]["m1"]["changed_input_fields"] == ["adapter"]
    for before, after in zip(read_jsonl(source / "m1.jsonl"), read_jsonl(output / "m1.jsonl"),
                             strict=True):
        assert {k: v for k, v in before.items() if k != "adapter"} == {
            k: v for k, v in after.items() if k != "adapter"}
        assert after["adapter"] == str(adapter.resolve())


def test_changed_source_after_inspection_blocks_all_writes(baselines):
    source, hashes, arguments, output = baselines
    plan = prepare_baseline_reuse(source, hashes, arguments)
    (source / "m1.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="after inspection"):
        write_baseline_reuse(plan)
    assert not output.exists()


def test_existing_conflict_is_preserved_before_any_other_copy(baselines):
    source, hashes, arguments, output = baselines
    output.mkdir()
    (output / "m1.jsonl").write_bytes(b"existing")
    with pytest.raises(ValueError, match="destination differs"):
        prepare_baseline_reuse(source, hashes, arguments)
    assert list(output.iterdir()) == [output / "m1.jsonl"]
    assert (output / "m1.jsonl").read_bytes() == b"existing"


def test_matching_copy_can_resume_without_touching_m2_m3(baselines):
    source, hashes, arguments, output = baselines
    output.mkdir()
    (output / "m0.jsonl").write_bytes((source / "m0.jsonl").read_bytes())
    before = (output / "m0.jsonl").stat().st_mtime_ns
    for arm in ("m2", "m3"):
        (output / f"{arm}.jsonl").write_bytes(b"LR-specific predictions")
    write_baseline_reuse(prepare_baseline_reuse(source, hashes, arguments))
    assert (output / "m0.jsonl").stat().st_mtime_ns == before
    for arm in ("m2", "m3"):
        assert (output / f"{arm}.jsonl").read_bytes() == b"LR-specific predictions"
