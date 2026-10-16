import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vqa_explanation.inference.export import export_predictions, read_jsonl
from vqa_explanation.inference.export_relocation import (
    prepare_relocation,
    prepare_snapshot_recovery,
    snapshot_export,
    verify_relocation_group,
    write_relocation,
)


class FakeRunner:
    def __init__(self, *args, **kwargs):
        pass

    def protocol_metadata(self, max_new_tokens):
        return {"max_new_tokens": max_new_tokens, "compute_dtype": "torch.float16"}

    def generate(self, *args, **kwargs):
        return "Đáp án: mèo\nGiải thích: Có mèo."


@pytest.fixture
def saved_export(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.jpg")
    row = dict(question_id="q1", image_id="im1", image_path="image.jpg",
               split="validation", question="Con gì?", answer="mèo", explanations=["Có mèo."])
    annotations = tmp_path / "data.jsonl"
    annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    for name in ("old-adapter", "current-adapter"):
        adapter = tmp_path / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"same weights")
    source = tmp_path / "original" / "m2.jsonl"
    args = SimpleNamespace(
        annotations=annotations, image_root=tmp_path, output=source,
        split="validation", limit=0, seed=42, model_id="model", revision="revision",
        max_new_tokens=128, min_pixels=200704, max_pixels=200704, load_in_4bit=True,
        adapter=tmp_path / "old-adapter", experiment_id="M2_SFT_CHOSEN",
    )
    export_predictions(args, runner_factory=FakeRunner)
    args.adapter = tmp_path / "current-adapter"
    args.output = tmp_path / "relocated" / "m2.jsonl"
    return source, args


def test_relocation_preserves_original_and_prediction_content(saved_export):
    source, args = saved_export
    original = source.read_bytes()
    plan = prepare_relocation(source, args)
    assert not args.output.parent.exists()
    receipt = write_relocation(plan)
    before, after = read_jsonl(source)[0], read_jsonl(args.output)[0]
    assert source.read_bytes() == original
    assert {k: v for k, v in before.items() if k != "adapter"} == {
        k: v for k, v in after.items() if k != "adapter"}
    assert after["adapter"] == str(args.adapter.resolve())
    assert receipt["changed_input_fields"] == ["adapter"]
    args.resume = True
    export_predictions(args, runner_factory=lambda *a, **kw: pytest.fail("complete export"))


@pytest.mark.parametrize("field,value", [("seed", 1), ("max_new_tokens", 64),
                                       ("revision", "other")])
def test_relocation_rejects_non_path_changes(saved_export, field, value):
    source, args = saved_export
    setattr(args, field, value)
    with pytest.raises(ValueError, match="more than storage"):
        prepare_relocation(source, args)
    assert not args.output.parent.exists()


def test_relocation_rejects_changed_weights(saved_export):
    source, args = saved_export
    (args.adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="more than storage"):
        prepare_relocation(source, args)


def test_relocation_rejects_corrupted_prediction(saved_export):
    source, args = saved_export
    rows = read_jsonl(source)
    rows[0]["prediction_answer"] = "wrong"
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parsed output changed"):
        prepare_relocation(source, args)


def test_relocation_rejects_existing_destination(saved_export):
    source, args = saved_export
    args.output.parent.mkdir()
    with pytest.raises(ValueError, match="new, separate"):
        prepare_relocation(source, args)


def test_manifest_without_file_does_not_invent_saved_rows(saved_export):
    source, args = saved_export
    source.unlink()
    plan = prepare_relocation(source, args)
    assert plan["receipt"]["source_file_present"] is False
    assert plan["receipt"]["copied_predictions"] == 0
    write_relocation(plan)
    assert read_jsonl(args.output) == []


def test_changed_source_after_inspection_blocks_copy(saved_export):
    source, args = saved_export
    plan = prepare_relocation(source, args)
    source.write_text(source.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after inspection"):
        write_relocation(plan)


def test_relocation_rejects_missing_manifest(saved_export):
    source, args = saved_export
    source.with_suffix(".jsonl.manifest.json").unlink()
    with pytest.raises(ValueError, match="without their manifest"):
        prepare_relocation(source, args)


def test_relocation_rejects_duplicate_id(saved_export):
    source, args = saved_export
    source.write_bytes(source.read_bytes() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_relocation(source, args)


def test_relocation_rejects_unterminated_line(saved_export):
    source, args = saved_export
    source.write_bytes(source.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="incomplete final line"):
        prepare_relocation(source, args)


def test_group_receipt_allows_resume_and_checks_original(saved_export):
    source, args = saved_export
    receipt = write_relocation(prepare_relocation(source, args))
    group = {"schema": "validation_export_relocation_v1",
             "source_dir": str(source.parent.resolve()), "receipts": [receipt]}
    (args.output.parent / "relocation_group.json").write_text(json.dumps(group), encoding="utf-8")
    assert verify_relocation_group(source.parent, args.output.parent) == group
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since relocation"):
        verify_relocation_group(source.parent, args.output.parent)


def test_relocation_rejects_child_of_original_directory(saved_export):
    source, args = saved_export
    args.output = source.parent / "child" / "m2.jsonl"
    with pytest.raises(ValueError, match="new, separate"):
        prepare_relocation(source, args)


def test_closed_snapshot_preserves_file_and_validates_rows(saved_export):
    source, args = saved_export
    records = read_jsonl(args.annotations)
    receipt = snapshot_export(source, records)
    snapshot = source.parent / "recovery/m2_0001/m2.jsonl"
    assert snapshot.read_bytes() == source.read_bytes()
    assert receipt["predictions"] == 1
    assert snapshot_export(source, records) is None


def test_snapshot_skips_uncommitted_last_line(saved_export):
    source, args = saved_export
    source.write_bytes(source.read_bytes().rstrip(b"\n"))
    assert snapshot_export(source, read_jsonl(args.annotations)) is None


def test_snapshot_rejects_bad_reference(saved_export):
    source, args = saved_export
    rows = read_jsonl(source)
    rows[0]["reference_answer"] = "other"
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference_answer changed"):
        snapshot_export(source, read_jsonl(args.annotations))


def test_recovery_uses_closed_snapshot_when_main_file_is_missing(saved_export):
    source, args = saved_export
    snapshot_export(source, read_jsonl(args.annotations))
    source.unlink()
    plan = prepare_snapshot_recovery(source, args)
    assert plan["receipt"]["copied_predictions"] == 1
    assert plan["source"].parent.name == "m2_0001"
    assert not args.output.parent.exists()
    write_relocation(plan)
    assert len(read_jsonl(args.output)) == 1
    assert not source.exists()


@pytest.mark.parametrize("changed", ["file", "receipt", "manifest", "count"])
def test_recovery_rejects_changed_closed_snapshot(saved_export, changed):
    source, args = saved_export
    snapshot_export(source, read_jsonl(args.annotations))
    folder = source.parent / "recovery/m2_0001"
    if changed == "file":
        (folder / source.name).write_text("{}\n")
    elif changed == "manifest":
        (folder / "m2.jsonl.manifest.json").write_text("{}")
    else:
        path = folder / "snapshot.json"
        receipt = json.loads(path.read_text())
        receipt["predictions" if changed == "count" else "output"] = 999
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="snapshot"):
        prepare_snapshot_recovery(source, args)
    assert not args.output.parent.exists()


def test_recovery_refuses_conflicting_main_prediction(saved_export):
    source, args = saved_export
    snapshot_export(source, read_jsonl(args.annotations))
    rows = read_jsonl(source)
    rows[0]["latency_seconds"] += 1
    source.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        prepare_snapshot_recovery(source, args)


def test_recovery_does_not_copy_over_existing_destination(saved_export):
    source, args = saved_export
    snapshot_export(source, read_jsonl(args.annotations))
    args.output.parent.mkdir()
    with pytest.raises(ValueError, match="new, separate"):
        prepare_snapshot_recovery(source, args)


def test_recovery_receipt_change_after_inspection_blocks_copy(saved_export):
    source, args = saved_export
    snapshot_export(source, read_jsonl(args.annotations))
    source.unlink()
    plan = prepare_snapshot_recovery(source, args)
    (plan["source"].parent / "snapshot.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed after inspection"):
        write_relocation(plan)
    assert not args.output.parent.exists()
