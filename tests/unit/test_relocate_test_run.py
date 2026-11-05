"""Storage relocation keeps partial test predictions and their source evidence."""

import json
import shutil
from pathlib import PurePosixPath

import pytest

from scripts.relocate_test_run import relocate_test_run
from tests.unit.test_study_test import factory, run
from tests.unit.test_study_test import inputs as inputs_fixture
from vqa_explanation.data.io import write_json
from vqa_explanation.inference.export import read_jsonl
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def partial(tmp_path):
    case = inputs_fixture.__wrapped__(tmp_path)
    case.output = tmp_path / "original/m3_seed42"
    runner, _ = factory(case, fail_after=27)
    with pytest.raises(RuntimeError, match="disconnection"):
        run(case, identity="m3_seed42", adapter=case.adapter, rubric=case.rubric,
            runner_factory=runner)
    relocated = tmp_path / "mounted"
    shutil.move(str(case.output.parent), relocated)
    case.source = relocated / "m3_seed42"
    case.output = tmp_path / "resumed/m3_seed42"
    other = tmp_path / "mounted_adapter"
    shutil.copytree(case.adapter, other)
    case.adapter = other
    return case


def relocate(case, **options):
    return relocate_test_run(
        case.environment, case.plan_path, case.plan_hash, "m3_seed42", case.source,
        case.output, adapter=case.adapter, rubric=case.rubric, **options)


def test_relocation_after_a_mount_change_preserves_source_and_resumes(partial):
    before = {p.relative_to(partial.source): sha256_file(p)
              for p in partial.source.rglob("*") if p.is_file()}
    preview = relocate(partial, inspect_only=True)
    assert preview["copied_predictions"] == 27 and not partial.output.exists()
    assert relocate(partial)["status"] == "relocated"
    old = read_jsonl(partial.source / "m3_seed42.jsonl")
    new = read_jsonl(partial.output / "m3_seed42.jsonl")
    assert [{k: v for k, v in r.items() if k != "adapter"} for r in old] == [
        {k: v for k, v in r.items() if k != "adapter"} for r in new]
    assert relocate(partial)["status"] == "verified"
    runner, generated = factory(partial)
    result = run(partial, identity="m3_seed42", adapter=partial.adapter,
                 rubric=partial.rubric, runner_factory=runner)
    assert result["status"] == "completed" and len(generated) == 1968 - 27
    assert relocate(partial)["status"] == "verified"
    assert before == {p.relative_to(partial.source): sha256_file(p)
                      for p in partial.source.rglob("*") if p.is_file()}


def test_changed_source_is_rejected_without_creating_a_destination(partial):
    output = partial.source / "m3_seed42.jsonl"
    output.write_bytes(output.read_bytes() + b"broken")
    with pytest.raises(ValueError):
        relocate(partial)
    assert not partial.output.exists()


def test_relocation_does_not_accept_changed_adapter_weights(partial):
    weights = partial.adapter / "adapter_model.safetensors"
    weights.write_bytes(b"other weights")
    with pytest.raises(ValueError):
        relocate(partial)
    assert not partial.output.exists()


def test_unverified_destination_and_modified_prefix_are_rejected(partial):
    partial.output.mkdir(parents=True)
    with pytest.raises(ValueError, match="matching run record"):
        relocate(partial)
    partial.output.rmdir()
    relocate(partial)
    output = partial.output / "m3_seed42.jsonl"
    output.write_bytes(b"changed" + output.read_bytes())
    with pytest.raises(ValueError, match="prefix changed"):
        relocate(partial)


def test_partial_main_behind_a_closed_snapshot_is_not_regenerated(partial):
    output = partial.source / "m3_seed42.jsonl"
    output.write_bytes(b"".join(output.read_bytes().splitlines(keepends=True)[:20]))
    with pytest.raises(ValueError, match="Closed snapshot"):
        relocate(partial)
    assert not partial.output.exists()


def test_changed_snapshot_stops_relocation(partial):
    snapshot = partial.source / "recovery/m3_seed42_0025/m3_seed42.jsonl"
    snapshot.write_bytes(snapshot.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="Closed snapshot"):
        relocate(partial)
    assert not partial.output.exists()


def test_linux_snapshot_locations_remain_valid_after_mount_relocation(partial):
    snapshot = partial.source / "recovery/m3_seed42_0025/snapshot.json"
    receipt = json.loads(snapshot.read_text("utf-8"))
    original = PurePosixPath("/content/drive/MyDrive/project/m3_seed42/m3_seed42.jsonl")
    receipt["output"] = str(original)
    receipt["snapshot"] = str(original.parent / "recovery/m3_seed42_0025/m3_seed42.jsonl")
    write_json(snapshot, receipt)
    assert relocate(partial, inspect_only=True)["copied_predictions"] == 27
