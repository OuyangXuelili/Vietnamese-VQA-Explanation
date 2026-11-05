"""CPU fixtures for test gating, protocol checks and interrupted exports."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from vqa_explanation.data.io import write_json, write_jsonl
from vqa_explanation.inference import study_test as final
from vqa_explanation.inference.export import read_jsonl
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def guidelines_inputs(inputs, tmp_path):
    from vqa_explanation.evaluation.guidelines import freeze_guidelines

    calibration, reference = tmp_path / "calibration", tmp_path / "reference"
    (calibration / "bundle/rater1").mkdir(parents=True)
    (calibration / "bundle/images").mkdir()
    (reference / "images").mkdir(parents=True)
    blank, rows = [], []
    for i in range(30):
        name = f"{i:04d}.jpg"
        image = calibration / "bundle/images" / name
        Image.new("RGB", (4, 4), "red").save(image)
        (reference / "images" / name).write_bytes(image.read_bytes())
        for slot in "ABCD":
            row = {"entry_id": f"{i}_{slot}", "image_index": i,
                   "question": "Màu gì?", "prediction_answer": "đỏ",
                   "prediction_explanation": "Vật màu đỏ.", "image_path": f"../images/{name}",
                   "review_status": "pending", "reviewer": None, "reviewed_at": None}
            blank.append(row)
            rows.append({**row, "image_path": f"images/{name}",
                         "assessment_type": "assistant_visual_review"})
    write_jsonl(calibration / "bundle/rater1/blank_ratings.jsonl", blank)
    (calibration / "rubric.yaml").write_bytes(
        (final.project_root() / "configs/evaluation/human_rubric_v1.yaml").read_bytes())
    write_json(calibration / "manifest.json", {"files": {
        p.relative_to(calibration).as_posix(): sha256_file(p)
        for p in calibration.rglob("*") if p.is_file()}})
    write_jsonl(reference / "reference_assessments.jsonl", rows)
    write_json(reference / "summary.json", {
        "assessment_method": "assistant_visual_review", "human_ratings": False,
        "split": "validation", "images": 30, "outputs": 120})
    write_json(reference / "manifest.json", {"files": {
        p.relative_to(reference).as_posix(): sha256_file(p)
        for p in reference.rglob("*") if p.is_file()}})
    inputs.plan["human_review"]["calibration_manifest_sha256"] = sha256_file(
        calibration / "manifest.json")
    write_json(inputs.plan_path, inputs.plan)
    inputs.plan_hash = sha256_file(inputs.plan_path)
    guides = final.project_root() / "docs/evaluation_guidelines.md"
    output = tmp_path / "frozen_instructions"
    result = freeze_guidelines(inputs.plan_path, inputs.plan_hash, calibration, reference,
                               guides, tmp_path / "data/test.jsonl", tmp_path / "test_runs", output)
    return SimpleNamespace(inputs=inputs, output=output, result=result, blank=blank,
                           calibration=calibration, reference=reference, guides=guides)


def test_instruction_freeze_is_explicit_and_does_not_create_human_ratings(guidelines_inputs):
    case = guidelines_inputs
    assert case.result["human_ratings"] == 0
    assert case.result["status"] == "instructions_frozen_human_calibration_pending"
    assert read_jsonl(case.calibration / "bundle/rater1/blank_ratings.jsonl") == case.blank
    plan_path = case.output / "plan.json"
    plan = final.read_test_plan(plan_path, case.result["plan_sha256"])
    for key in ("runs", "inference_protocol", "metrics", "human_review"):
        assert plan.get(key) == case.inputs.plan.get(key)
    report = final.run_test(case.inputs.environment, plan_path, case.result["plan_sha256"],
                            "m0", case.inputs.output, rubric=case.output, inspect_only=True)
    assert report["status"] == "inputs_verified"
    assert not case.inputs.output.exists()
    lock = json.loads((case.output / "lock.json").read_text("utf-8"))
    assert lock["human_calibration_complete"] is False and lock["human_raters"] == 0
    with pytest.raises(ValueError, match="frozen rubric"):
        final.verify_rubric(case.output, case.inputs.plan)


@pytest.mark.parametrize("name", ["guidelines.md", "rubric.yaml", "reference_summary.json",
                                  "reference_assessments.jsonl", "parent_plan.json", "plan.json",
                                  "calibration_manifest.json", "reference_manifest.json"])
def test_instruction_freeze_rejects_changed_files(guidelines_inputs, name):
    from vqa_explanation.evaluation.guidelines import verify_frozen_guidelines

    case = guidelines_inputs
    plan = json.loads((case.output / "plan.json").read_text("utf-8"))
    path = case.output / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        verify_frozen_guidelines(case.output, plan)


@pytest.mark.parametrize("name", ["guidelines.md", "reference_assessments.jsonl"])
def test_rehashing_lock_cannot_replace_instructions_or_reference(guidelines_inputs, name):
    from vqa_explanation.evaluation.guidelines import verify_frozen_guidelines

    case = guidelines_inputs
    plan = json.loads((case.output / "plan.json").read_text("utf-8"))
    path = case.output / name
    path.write_bytes(path.read_bytes() + b"\n")
    lock = json.loads((case.output / "lock.json").read_text("utf-8"))
    lock["files"][name] = sha256_file(path)
    write_json(case.output / "lock.json", lock)
    with pytest.raises(ValueError):
        verify_frozen_guidelines(case.output, plan)


def test_amendment_cannot_change_a_model_identity(guidelines_inputs):
    from vqa_explanation.evaluation.guidelines import verify_frozen_guidelines

    case = guidelines_inputs
    plan = json.loads((case.output / "plan.json").read_text("utf-8"))
    plan["runs"]["m3_seed42"]["training_seed"] = 99
    write_json(case.output / "plan.json", plan)
    lock = json.loads((case.output / "lock.json").read_text("utf-8"))
    lock["files"]["plan.json"] = sha256_file(case.output / "plan.json")
    write_json(case.output / "lock.json", lock)
    with pytest.raises(ValueError, match="sequencing only"):
        verify_frozen_guidelines(case.output, plan)


def test_instructions_cannot_be_amended_after_a_test_export(guidelines_inputs, tmp_path):
    from vqa_explanation.evaluation.guidelines import freeze_guidelines

    case = guidelines_inputs
    runs = tmp_path / "test_runs"
    runs.mkdir()
    (runs / "other_export.jsonl").write_text('{}\n', "utf-8")
    with pytest.raises(ValueError, match="before any test export"):
        freeze_guidelines(case.inputs.plan_path, case.inputs.plan_hash, case.calibration,
                          case.reference, case.guides, tmp_path / "data/test.jsonl", runs,
                          tmp_path / "second_freeze")


def test_progress_preserves_pending_human_calibration(guidelines_inputs):
    from vqa_explanation.evaluation.test_report import build_test_report

    case = guidelines_inputs
    report = build_test_report(
        case.output / "plan.json", case.result["plan_sha256"],
        case.inputs.environment.parent / "data/test.jsonl",
        {name: case.inputs.output / name for name in final.IDENTITIES}, inspect_only=True)
    assert report["human_calibration_complete_before_inference"] is False
    assert report["human_evaluation_complete"] is False
    assert report["metrics_computed"] is False


def test_test_notebook_is_thin_and_stops_on_blocked_inspection(tmp_path):
    from scripts.package_source import source_files

    root = final.project_root()
    path = root / "notebooks/08_evaluate_test.ipynb"
    notebook = json.loads(path.read_text("utf-8"))
    codes = {c["id"]: "".join(c["source"]) for c in notebook["cells"]
             if c["cell_type"] == "code"}
    assert len(codes) == 7 and path in source_files(root)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")
    namespace = {"plan": {"runs": {"m0": {"adapter_at_validation": None}}},
                 "RUN_IDENTITIES": None, "RUN_DIR_NAMES": {}, "RELOCATE_IDENTITIES": [],
                 "json": json, "PYTHON": "python", "REPO": tmp_path,
                 "ENVIRONMENT": tmp_path / "env.yaml", "PLAN": tmp_path / "plan.json",
                 "launch": {"plan_sha256": "a" * 64}, "OUTPUT": tmp_path / "out",
                 "DATA": tmp_path, "subprocess": SimpleNamespace(
                     check_output=lambda *a, **kw: '{"status":"blocked"}')}
    with pytest.raises(RuntimeError, match="input blocked"):
        exec(codes["test-inspection"], namespace)
    assert namespace["INSPECTED"] is False and namespace["COMMANDS"] == []


def test_amended_export_keeps_plan_identity_when_interrupted(guidelines_inputs):
    case = guidelines_inputs
    runner, generated = factory(case.inputs, fail_after=3)
    with pytest.raises(RuntimeError, match="disconnection"):
        final.run_test(case.inputs.environment, case.output / "plan.json",
                       case.result["plan_sha256"], "m0", case.inputs.output,
                       rubric=case.output, runner_factory=runner)
    assert len(generated) == 3
    assert len(read_jsonl(case.inputs.output / "m0.jsonl")) == 3
    record = json.loads((case.inputs.output / "test_run.json").read_text("utf-8"))
    assert record["plan_sha256"] == case.result["plan_sha256"]
    assert record["rubric_lock_sha256"] == case.result["lock_sha256"]
    assert not (case.inputs.output / "completion.json").exists()


def test_progress_does_not_infer_human_calibration_from_an_old_plan(inputs):
    from vqa_explanation.evaluation.test_report import build_test_report

    report = build_test_report(
        inputs.plan_path, inputs.plan_hash, inputs.environment.parent / "data/test.jsonl",
        {name: inputs.output / name for name in final.IDENTITIES}, inspect_only=True)
    assert report["human_calibration_complete_before_inference"] is None
    assert report["human_evaluation_complete"] is False


@pytest.fixture
def inputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name, color in (("a.jpg", "red"), ("b.jpg", "blue")):
        Image.new("RGB", (4, 4), color).save(images / name)
    rows = [{"question_id": f"q{i}", "image_id": f"im{i % 2}",
             "image_path": "a.jpg" if i % 2 == 0 else "b.jpg", "split": "test",
             "question": "Màu gì?", "answer": "đỏ", "explanations": ["Vật có màu đỏ."]}
            for i in range(1968)]
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "all.jsonl", rows)
    write_jsonl(data / "test.jsonl", rows)
    checksums = data / "checksums.csv"
    checksums.write_text("image_name,size_bytes,sha256\n" + "".join(
        f"{p.name},{p.stat().st_size},{sha256_file(p)}\n" for p in images.iterdir()), "utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({
        "annotations": str(data / "all.jsonl"), "image_root": str(images),
        "run_root": str(tmp_path / "training"), "image_checksums": str(checksums)}), "utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{}', "utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
    protocol = {"model_id": final.MODEL_ID, "model_revision": final.MODEL_REVISION,
                "seed": 42, "min_pixels": 200704, "max_pixels": 200704,
                "max_new_tokens": 128, "prompt_sha256": "a" * 64,
                "generation_config": {"do_sample": False, "max_new_tokens": 128},
                "load_in_4bit": True, "compute_dtype": "torch.float16"}
    source_files = ["scripts/run_m0.py", "pyproject.toml",
                    "src/vqa_explanation/inference/export.py",
                    "src/vqa_explanation/inference/prompting.py",
                    "src/vqa_explanation/inference/qwen25_vl.py",
                    "src/vqa_explanation/inference/generation_safety.py",
                    "src/vqa_explanation/training/sft.py",
                    "src/vqa_explanation/utils/model_loading.py"]
    plan = {"schema": "final_test_preparation_v1", "split": "test", "questions_per_run": 1968,
            "test_images": 2, "expected_predictions": 15744, "inference_order_seed": 42,
            "runs": {name: {"experiment_id": final.ARMS[arm], "training_seed": seed,
                            "adapter_files": None if arm == "m0" else {
                                p.name: sha256_file(p) for p in adapter.iterdir()}}
                     for name, (arm, seed) in final.IDENTITIES.items()},
            "inference_protocol": protocol,
            "inference_source": {name: sha256_file(final.project_root() / name)
                                 for name in source_files},
            "annotations_sha256": sha256_file(data / "all.jsonl"),
            "test_annotations_sha256": sha256_file(data / "test.jsonl"),
            "human_review": {"calibration_manifest_sha256": "c" * 64}}
    plans = tmp_path / "plans"
    plans.mkdir()
    plan_path = plans / "plan.json"
    write_json(plan_path, plan)
    rubric = tmp_path / "rubric"
    rubric.mkdir()
    (rubric / "rubric.yaml").write_bytes(
        (final.project_root() / "configs/evaluation/human_rubric_v1.yaml").read_bytes())
    summary = {"status": "calibration_complete", "split": "validation",
               "raters": {"rater1": {arm: {"n": 30} for arm in final.ARMS.values()}}}
    write_json(rubric / "calibration_summary.json", summary)
    write_json(rubric / "lock.json", {
        "schema": "frozen_human_rubric_v1", "status": "frozen",
        "calibration_manifest_sha256": "c" * 64,
        "rubric_sha256": sha256_file(rubric / "rubric.yaml"),
        "summary_sha256": sha256_file(rubric / "calibration_summary.json"),
        "raters": 1, "ratings": {"rater1": {"sha256": "d" * 64}}})
    return SimpleNamespace(environment=environment, plan_path=plan_path,
                           plan_hash=sha256_file(plan_path), plan=plan, rubric=rubric,
                           adapter=adapter, output=tmp_path / "evaluation", rows=rows)


def run(inputs, **options):
    return final.run_test(inputs.environment, inputs.plan_path, inputs.plan_hash,
                          options.pop("identity", "m0"), inputs.output, **options)


def factory(inputs, *, fail_after=None, changed_protocol=False):
    generated = []

    class Runner:
        def __init__(self, *a, **kw):
            pass

        def protocol_metadata(self, _):
            result = {key: value for key, value in inputs.plan["inference_protocol"].items()
                      if key not in {"model_id", "model_revision", "seed"}}
            if changed_protocol:
                result["compute_dtype"] = "torch.bfloat16"
            return result

        def generate(self, *a, **kw):
            if fail_after is not None and len(generated) == fail_after:
                raise RuntimeError("fixture disconnection")
            generated.append(a)
            return "Đáp án: đỏ\nGiải thích: Vật có màu đỏ."

    return Runner, generated


def test_inspection_reports_missing_rubric_and_does_not_create_outputs(inputs):
    report = run(inputs, inspect_only=True)
    assert report["status"] == "blocked" and len(report["blockers"]) == 1
    assert report["runtime_verified_in_this_step"] is False
    assert not inputs.output.exists()
    with pytest.raises(ValueError, match="freeze the rubric"):
        run(inputs)
    assert not inputs.output.exists()


def test_frozen_inputs_inspect_without_gpu_or_predictions(inputs):
    report = run(inputs, rubric=inputs.rubric, inspect_only=True)
    assert report["status"] == "inputs_verified"
    assert report["test_predictions_generated"] is False
    assert not inputs.output.exists()


@pytest.mark.parametrize("identity", list(final.IDENTITIES))
def test_identity_selection_has_explicit_adapter_requirements(inputs, identity):
    report = run(inputs, identity=identity, rubric=inputs.rubric, inspect_only=True)
    assert bool(report["blockers"]) == (identity != "m0")
    if identity != "m0":
        assert run(inputs, identity=identity, rubric=inputs.rubric,
                   adapter=inputs.adapter, inspect_only=True)["status"] == "inputs_verified"


@pytest.mark.parametrize("fault", ["plan", "annotations", "images", "adapter", "rubric"])
def test_changed_inputs_stop_before_any_output(inputs, fault):
    path = {"plan": inputs.plan_path, "annotations": inputs.environment.parent / "data/test.jsonl",
            "images": inputs.environment.parent / "images/a.jpg",
            "adapter": inputs.adapter / "adapter_model.safetensors",
            "rubric": inputs.rubric / "rubric.yaml"}[fault]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        run(inputs, identity="m1", adapter=inputs.adapter, rubric=inputs.rubric)
    assert not inputs.output.exists()


def test_unplanned_seed_and_m0_adapter_are_rejected(inputs):
    with pytest.raises(ValueError, match="Unknown test identity"):
        run(inputs, identity="m3_seed45", inspect_only=True)
    with pytest.raises(ValueError, match="must not load an adapter"):
        run(inputs, adapter=inputs.adapter, inspect_only=True)


def test_changed_runtime_stops_before_first_prediction(inputs):
    runner, generated = factory(inputs, changed_protocol=True)
    with pytest.raises(ValueError, match="protocol differs"):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    assert generated == []
    assert not (inputs.output / "m0.jsonl").exists()
    assert not (inputs.output / "completion.json").exists()


def test_interrupted_export_resumes_only_missing_rows_and_keeps_snapshots(inputs):
    runner, generated = factory(inputs, fail_after=27)
    with pytest.raises(RuntimeError, match="disconnection"):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    output = inputs.output / "m0.jsonl"
    prefix = output.read_bytes()
    snapshot = inputs.output / "recovery/m0_0025/m0.jsonl"
    snapshot_hash = sha256_file(snapshot)
    assert len(generated) == 27 and len(read_jsonl(output)) == 27
    runner, resumed = factory(inputs)
    result = run(inputs, rubric=inputs.rubric, runner_factory=runner)
    assert result["status"] == "completed" and len(resumed) == 1968 - 27
    assert result["test_predictions_generated"] is True
    assert output.read_bytes().startswith(prefix) and sha256_file(snapshot) == snapshot_hash
    assert len(read_jsonl(output)) == 1968
    before = output.read_bytes()

    def no_model(*a, **kw):
        pytest.fail("A completed export must not load a model")

    result = run(inputs, rubric=inputs.rubric, runner_factory=no_model)
    assert result["test_predictions_generated"] is False
    assert result["runtime_verified_in_this_step"] is False
    assert output.read_bytes() == before


def test_missing_main_rows_with_a_closed_snapshot_are_not_regenerated(inputs):
    runner, _ = factory(inputs, fail_after=27)
    with pytest.raises(RuntimeError):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    output = inputs.output / "m0.jsonl"
    output.write_bytes(b"")
    runner, generated = factory(inputs)
    with pytest.raises(ValueError, match="Recovery required"):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    assert generated == [] and output.read_bytes() == b""


def test_resume_cannot_silently_switch_training_seed(inputs):
    runner, _ = factory(inputs, fail_after=1)
    with pytest.raises(RuntimeError):
        run(inputs, identity="m2_seed42", adapter=inputs.adapter,
            rubric=inputs.rubric, runner_factory=runner)
    with pytest.raises(ValueError, match="Test run inputs changed"):
        run(inputs, identity="m2_seed43", adapter=inputs.adapter, rubric=inputs.rubric)


def test_validation_source_fingerprint_is_not_changed_by_test_entry_point(inputs):
    original = copy.deepcopy(inputs.plan["inference_source"])
    run(inputs, rubric=inputs.rubric, inspect_only=True)
    assert {name: sha256_file(final.project_root() / name) for name in original} == original


def test_damaged_main_export_can_recover_a_closed_prefix_into_a_new_run(inputs):
    runner, _ = factory(inputs, fail_after=27)
    with pytest.raises(RuntimeError):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    damaged = inputs.output / "m0.jsonl"
    damaged.write_bytes(b'{"truncated')
    snapshot = inputs.output / "recovery/m0_0025/m0.jsonl"
    source_hash = sha256_file(snapshot)
    inputs.output = inputs.output.parent / "recovered"
    preview = run(inputs, rubric=inputs.rubric, recover_from=snapshot, inspect_only=True)
    assert preview["recovery"]["source_sha256"] == source_hash
    assert not inputs.output.exists()
    result = run(inputs, rubric=inputs.rubric, recover_from=snapshot)
    assert result["status"] == "recovered" and result["test_predictions_generated"] is False
    assert len(read_jsonl(inputs.output / "m0.jsonl")) == 25
    assert damaged.read_bytes() == b'{"truncated' and sha256_file(snapshot) == source_hash
    runner, resumed = factory(inputs)
    assert run(inputs, rubric=inputs.rubric, runner_factory=runner)["status"] == "completed"
    assert len(resumed) == 1968 - 25


def test_changed_snapshot_is_not_copied_or_regenerated(inputs):
    runner, _ = factory(inputs, fail_after=27)
    with pytest.raises(RuntimeError):
        run(inputs, rubric=inputs.rubric, runner_factory=runner)
    snapshot = inputs.output / "recovery/m0_0025/m0.jsonl"
    snapshot.write_bytes(snapshot.read_bytes() + b"changed")
    inputs.output = inputs.output.parent / "recovered"
    with pytest.raises(ValueError, match="snapshot receipt or file changed"):
        run(inputs, rubric=inputs.rubric, recover_from=snapshot)
    assert not inputs.output.exists()


def test_rubric_created_by_the_real_freezer_passes_the_test_gate(inputs, tmp_path):
    from tests.unit.test_rubric_calibration import RUBRIC, filled_ratings, practice
    from vqa_explanation.evaluation.rubric_calibration import freeze_rubric

    folder = tmp_path / "calibration_fixture"
    folder.mkdir()
    calibration = practice.__wrapped__(folder)
    ratings = filled_ratings(calibration)
    frozen = tmp_path / "real_frozen_rubric"
    freeze_rubric(calibration, ratings, RUBRIC, frozen)
    plan = copy.deepcopy(inputs.plan)
    plan["human_review"]["calibration_manifest_sha256"] = sha256_file(
        calibration / "manifest.json")
    assert final.verify_rubric(frozen, plan) == sha256_file(frozen / "lock.json")


def test_inspection_rejects_a_run_from_another_seed(inputs):
    runner, _ = factory(inputs, fail_after=1)
    with pytest.raises(RuntimeError):
        run(inputs, identity="m2_seed42", adapter=inputs.adapter,
            rubric=inputs.rubric, runner_factory=runner)
    with pytest.raises(ValueError, match="Test run inputs changed"):
        run(inputs, identity="m2_seed43", adapter=inputs.adapter,
            rubric=inputs.rubric, inspect_only=True)


@pytest.mark.parametrize("fault", ["predictions", "manifest", "receipt", "count"])
def test_closed_export_changes_stop_before_loading_a_model(inputs, fault):
    runner, _ = factory(inputs)
    run(inputs, rubric=inputs.rubric, runner_factory=runner)
    paths = {"predictions": inputs.output / "m0.jsonl",
             "manifest": inputs.output / "m0.jsonl.manifest.json",
             "receipt": inputs.output / "test_run.json",
             "count": inputs.output / "completion.json"}
    path = paths[fault]
    if fault == "count":
        record = json.loads(path.read_text("utf-8"))
        record["predictions"] = 1967
        write_json(path, record)
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    changed = path.read_bytes()

    def no_model(*a, **kw):
        pytest.fail("A changed closed export must not load a model")

    with pytest.raises(ValueError, match="Completed test export changed|Test run inputs changed"):
        run(inputs, rubric=inputs.rubric, runner_factory=no_model)
    assert path.read_bytes() == changed


def notebook_project_path(root):
    notebook = json.loads((final.project_root() / "notebooks/08_evaluate_test.ipynb")
                          .read_text("utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell.get("id") == "test-restore")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "project_path")
    namespace = {"ROOT": root, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"), namespace)
    return namespace["project_path"]


def map_drive_shortcuts(monkeypatch, aliases):
    resolve = Path.resolve

    def mapped(path, *args, **kwargs):
        for alias, target in sorted(aliases.items(), key=lambda item: -len(item[0].parts)):
            if path.is_relative_to(alias):
                return resolve(target / path.relative_to(alias), *args, **kwargs)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mapped)


def test_notebook_accepts_separately_mounted_project_folders(tmp_path, monkeypatch):
    root, shared = tmp_path / "project", tmp_path / "shared_source"
    map_drive_shortcuts(monkeypatch, {root / "source": shared})
    path = notebook_project_path(root)
    assert path("source/releases/code.zip") == shared / "releases/code.zip"
    assert path("datasets/test.zip") == root / "datasets/test.zip"


def test_notebook_rejects_paths_outside_declared_project_folders(tmp_path):
    path = notebook_project_path(tmp_path / "project")
    for value in ("", "../other", "source/../other", "unknown/file", str(tmp_path / "file")):
        with pytest.raises(ValueError, match="Project path"):
            path(value)


def test_notebook_rejects_a_shortcut_escaping_its_project_component(tmp_path, monkeypatch):
    root = tmp_path / "project"
    map_drive_shortcuts(monkeypatch, {
        root / "source": tmp_path / "shared_source",
        root / "source/releases/escape.zip": tmp_path / "other/escape.zip",
    })
    with pytest.raises(ValueError, match="Project path"):
        notebook_project_path(root)("source/releases/escape.zip")
