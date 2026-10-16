import hashlib
import json
import pickle
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.resume_pinned_m1 import (
    EPOCH_BUDGET,
    EPOCH_STEPS,
    EXTENDED_BUDGET,
    main,
    resume_run_root,
    training_command,
    validate_stop,
    verify_checkpoint_payloads,
    verify_epoch_completion,
    verify_no_followup,
    verify_source,
)

M1_NOTEBOOK = "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb"


def cell_source(notebook, cell_id):
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    assert cell["cell_type"] == "code"
    return "".join(cell["source"])


def test_source_hash_and_tampering(tmp_path):
    source = tmp_path / "train.py"
    source.write_text("original")
    manifest = {"train.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_source(tmp_path, manifest)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Pinned source changed"):
        verify_source(tmp_path, manifest)


def test_source_cannot_escape_release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (tmp_path / "outside.py").write_text("outside")
    with pytest.raises(ValueError, match="escapes"):
        verify_source(root, {"../outside.py": "unused"})


@pytest.mark.parametrize("stop", [600, 500, 650, 1100, 3700, -1, True])
def test_invalid_stop(stop):
    with pytest.raises(ValueError):
        validate_stop({"global_step": 600, "max_steps": 3683, "save_steps": 100}, stop)


def test_resume_to_800():
    validate_stop({"global_step": 600, "max_steps": 3683, "save_steps": 100}, 800)


def test_resume_to_1000():
    validate_stop({"global_step": 800, "max_steps": 3683, "save_steps": 100}, 1000)
    validate_stop({"global_step": 900, "max_steps": 3683, "save_steps": 100}, 1000)


@pytest.mark.parametrize("start,stop", [(1000, 1100), (1000, 1200), (1100, 1200)])
def test_bounded_extension(start, stop):
    validate_stop({"global_step": start, "max_steps": 3683, "save_steps": 100},
                  stop, budget_protocol=EXTENDED_BUDGET)


@pytest.mark.parametrize("start,stop", [(900, 1200), (1000, 1250), (1000, 1300),
                                        (1000, 1000), (1200, 1200)])
def test_extension_rejects_wrong_start_or_stop(start, stop):
    with pytest.raises(ValueError):
        validate_stop({"global_step": start, "max_steps": 3683, "save_steps": 100},
                      stop, budget_protocol=EXTENDED_BUDGET)


def test_extension_requires_explicit_protocol():
    state = {"global_step": 1000, "max_steps": 3683, "save_steps": 100}
    with pytest.raises(ValueError):
        validate_stop(state, 1200)
    with pytest.raises(ValueError, match="Unknown"):
        validate_stop(state, 1200, budget_protocol="unbounded")


def epoch_state(step=1200):
    return {"global_step": step, "max_steps": EPOCH_STEPS, "num_train_epochs": 1,
            "save_steps": 100, "eval_steps": 100}


@pytest.mark.parametrize("step", [1200, 2500, 2800, 3600])
def test_epoch_finish_uses_the_original_schedule(step):
    assert validate_stop(epoch_state(step), EPOCH_STEPS, budget_protocol=EPOCH_BUDGET) is None


def test_epoch_extension_can_pause_on_an_intermediate_checkpoint():
    assert validate_stop(epoch_state(), 1400, budget_protocol=EPOCH_BUDGET) == 1400


@pytest.mark.parametrize("change", [
    {"global_step": 1100}, {"max_steps": 3682}, {"num_train_epochs": 2},
    {"num_train_epochs": True}, {"eval_steps": 200}, {"save_steps": 200},
])
def test_epoch_extension_rejects_a_changed_schedule(change):
    with pytest.raises(ValueError, match="original one-epoch schedule"):
        validate_stop(epoch_state() | change, EPOCH_STEPS, budget_protocol=EPOCH_BUDGET)


@pytest.mark.parametrize("stop", [1200, 1250, 3684, True])
def test_epoch_extension_rejects_invalid_targets(stop):
    with pytest.raises(ValueError):
        validate_stop(epoch_state(), stop, budget_protocol=EPOCH_BUDGET)


def test_epoch_extension_requires_explicit_authorization():
    for protocol in ("vqa_nle_budget_v2", EXTENDED_BUDGET):
        with pytest.raises(ValueError):
            validate_stop(epoch_state(), EPOCH_STEPS, budget_protocol=protocol)


@pytest.mark.parametrize("stop", [None, 1400])
def test_training_command_preserves_schedule_and_only_pauses_before_the_end(tmp_path, stop):
    paths = {key: tmp_path / key for key in ("annotations", "image_root", "image_checksums")}
    command = training_command(tmp_path / "source", paths, tmp_path / "run",
                               tmp_path / "checkpoint-1200", stop)
    assert "--resume-from-checkpoint" in command
    assert "--max-steps" not in command and "--num-train-epochs" not in command
    assert "--learning-rate" not in command
    if stop is None:
        assert "--stop-after-step" not in command
    else:
        assert command[-2:] == ["--stop-after-step", "1400"]


@pytest.fixture
def completed_epoch(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    run = tmp_path / "sft-full_epoch"
    training = run / "training"
    best = training / "checkpoint-3500"
    adapter = training / "adapter"
    best.mkdir(parents=True)
    adapter.mkdir()
    weights = {"base.layer.lora_A.weight": torch.ones(2, 4)}
    save_file(weights, str(best / "adapter_model.safetensors"))
    save_file(weights, str(adapter / "adapter_model.safetensors"))
    (best / "trainer_state.json").write_text(json.dumps({"global_step": 3500}))
    (training / "trainer_state.json").write_text(json.dumps({
        "epoch": 1.0, "global_step": EPOCH_STEPS, "max_steps": EPOCH_STEPS,
        "best_model_checkpoint": str(best), "best_metric": 1.0,
    }))
    (training / "training_result.json").write_text(json.dumps({
        "status": "passed", "global_step": EPOCH_STEPS, "target_global_step": EPOCH_STEPS,
        "best_model_checkpoint": str(best), "best_eval_loss": 1.0,
        "final_validation": {"eval_loss": 1.0},
    }))
    return run


def test_epoch_completion_distinguishes_last_and_best(completed_epoch):
    report = verify_epoch_completion(completed_epoch, EPOCH_STEPS)
    assert report["epoch"] == 1.0 and report["final_optimizer_step"] == EPOCH_STEPS
    assert report["selected_step"] == 3500
    assert report["selected_adapter_sha256"] == report["selected_checkpoint_sha256"]


@pytest.mark.parametrize("fault", ["epoch", "step", "target", "path", "loss", "weights"])
def test_epoch_completion_rejects_incomplete_or_wrong_selection(completed_epoch, fault):
    import torch
    from safetensors.torch import save_file

    training = completed_epoch / "training"
    path = training / ("trainer_state.json" if fault in ("epoch", "step")
                       else "training_result.json")
    content = json.loads(path.read_text())
    if fault == "epoch":
        content["epoch"] = 0.999
    elif fault == "step":
        content["global_step"] = 3600
    elif fault == "target":
        content["target_global_step"] = 3600
    elif fault == "path":
        content["best_model_checkpoint"] = "wrong checkpoint"
    elif fault == "loss":
        content["final_validation"]["eval_loss"] = float("nan")
    elif fault == "weights":
        save_file({"base.layer.lora_A.weight": torch.zeros(2, 4)},
                  str(training / "adapter/adapter_model.safetensors"))
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        verify_epoch_completion(completed_epoch, EPOCH_STEPS)


@pytest.fixture
def checkpoint_900(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint = tmp_path / "training_local/sft-full_20261003_131435_235939/training/checkpoint-900"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "run_metadata.json").write_text(json.dumps({
        "compute_dtype": "torch.bfloat16",
    }))
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 900, "max_steps": 3683, "save_steps": 100,
        "log_history": [{"step": 900, "eval_loss": 1.1}],
    }))
    (checkpoint / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "bias": "none", "r": 2, "rank_pattern": {},
    }))
    weights = {
        "base_model.model.layer.lora_A.weight": torch.ones(2, 4),
        "base_model.model.layer.lora_B.weight": torch.ones(3, 2),
    }
    save_file(weights, str(checkpoint / "adapter_model.safetensors"))
    parameters = [torch.nn.Parameter(weight.clone()) for weight in weights.values()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for moment in optimizer.state.values():
        moment["step"].fill_(900)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0).state_dict()
    scheduler.update(last_epoch=900, _step_count=901)
    torch.save(scheduler, checkpoint / "scheduler.pt")
    torch.save({
        "python": random.Random(42).getstate(),
        "numpy": np.random.RandomState(42).get_state(),
        "cpu": torch.Generator().manual_seed(42).get_state(),
        "cuda": torch.ones(16, dtype=torch.uint8),
    }, checkpoint / "rng_state.pth")
    # This file is deliberately opaque: the audit must hash, not unpickle it.
    (checkpoint / "training_args.bin").write_bytes(b"opaque training arguments")
    return checkpoint


@pytest.fixture
def notebook_checkpoint_1000(checkpoint_900):
    import torch

    project = checkpoint_900.parents[3]
    checkpoint = project / "runs/m1/sft-full_20261003_175915_354865/training/checkpoint-1000"
    shutil.copytree(checkpoint_900, checkpoint)
    shutil.copyfile(checkpoint_900.parent / "run_metadata.json",
                    checkpoint.parent / "run_metadata.json")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    state["global_step"] = 1000
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    for moment in optimizer["state"].values():
        moment["step"].fill_(1000)
    torch.save(optimizer, checkpoint / "optimizer.pt")
    scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
    scheduler.update(last_epoch=1000, _step_count=1001)
    torch.save(scheduler, checkpoint / "scheduler.pt")
    return checkpoint


def test_checkpoint_1000_payload(notebook_checkpoint_1000):
    report = verify_checkpoint_payloads(notebook_checkpoint_1000)
    assert report["global_step"] == report["scheduler_step"] == 1000


@pytest.fixture
def notebook_checkpoint_1100(notebook_checkpoint_1000):
    import torch

    project = notebook_checkpoint_1000.parents[4]
    checkpoint = project / "runs/m1/sft-full_20261004_065122_901248/training/checkpoint-1100"
    shutil.copytree(notebook_checkpoint_1000, checkpoint)
    shutil.copyfile(notebook_checkpoint_1000.parent / "run_metadata.json",
                    checkpoint.parent / "run_metadata.json")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    state["global_step"] = 1100
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    for moment in optimizer["state"].values():
        moment["step"].fill_(1100)
    torch.save(optimizer, checkpoint / "optimizer.pt")
    scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
    scheduler.update(last_epoch=1100, _step_count=1101)
    torch.save(scheduler, checkpoint / "scheduler.pt")
    return checkpoint


def test_checkpoint_1100_payload(notebook_checkpoint_1100):
    report = verify_checkpoint_payloads(notebook_checkpoint_1100)
    assert report["global_step"] == report["scheduler_step"] == 1100


def test_checkpoint_payloads_are_read_only(checkpoint_900):
    import numpy as np
    import torch

    before = {file.name: file.read_bytes() for file in checkpoint_900.iterdir()}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    safe_globals = frozenset(torch.serialization.get_safe_globals())
    report = verify_checkpoint_payloads(checkpoint_900)
    assert report["global_step"] == report["scheduler_step"] == 900
    assert report["adapter_tensors"] == report["optimizer_parameters"] == 2
    assert report["adapter_parameters"] == 14
    assert report["cuda_rng_format"] == "tensor"
    assert report["training_args_policy"] == "hashed_without_unpickling"
    assert before == {file.name: file.read_bytes() for file in checkpoint_900.iterdir()}
    assert python_state == random.getstate()
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert safe_globals == frozenset(torch.serialization.get_safe_globals())


@pytest.mark.parametrize("fault", [
    "trainer_step", "adapter_nan", "rank", "optimizer_step", "optimizer_nan",
    "optimizer_shape", "negative_second_moment", "scheduler_step", "scheduler_lr",
    "missing_rng", "cuda_rng", "fp16_missing_scaler",
])
def test_checkpoint_payload_corruption_is_rejected(checkpoint_900, fault):
    import torch
    from safetensors.torch import load_file, save_file

    checkpoint = checkpoint_900
    if fault == "trainer_step":
        path = checkpoint / "trainer_state.json"
        state = json.loads(path.read_text())
        state["global_step"] = 800
        path.write_text(json.dumps(state))
    elif fault in {"adapter_nan", "rank"}:
        path = checkpoint / "adapter_model.safetensors"
        weights = load_file(str(path))
        if fault == "adapter_nan":
            next(iter(weights.values()))[0, 0] = float("nan")
        else:
            weights["base_model.model.layer.lora_A.weight"] = torch.ones(1, 4)
        save_file(weights, str(path))
    elif fault.startswith("optimizer_") or fault == "negative_second_moment":
        path = checkpoint / "optimizer.pt"
        optimizer = torch.load(path, weights_only=True)
        moment = next(iter(optimizer["state"].values()))
        if fault == "optimizer_step":
            moment["step"].fill_(800)
        elif fault == "optimizer_nan":
            moment["exp_avg"][0, 0] = float("inf")
        elif fault == "optimizer_shape":
            moment["exp_avg"] = moment["exp_avg_sq"] = torch.ones(1, 1)
        else:
            moment["exp_avg_sq"][0, 0] = -1
        torch.save(optimizer, path)
    elif fault.startswith("scheduler_"):
        path = checkpoint / "scheduler.pt"
        scheduler = torch.load(path, weights_only=True)
        scheduler["last_epoch" if fault == "scheduler_step" else "_last_lr"] = (
            800 if fault == "scheduler_step" else [0.05]
        )
        torch.save(scheduler, path)
    elif fault == "missing_rng":
        (checkpoint / "rng_state.pth").unlink()
    elif fault == "cuda_rng":
        torch.save({"python": random.Random(42).getstate(),
                    "numpy": __import__("numpy").random.RandomState(42).get_state(),
                    "cpu": torch.Generator().get_state(), "cuda": []},
                   checkpoint / "rng_state.pth")
    else:
        (checkpoint.parent / "run_metadata.json").write_text(json.dumps({
            "compute_dtype": "torch.float16",
        }))
    with pytest.raises(ValueError):
        verify_checkpoint_payloads(checkpoint)


def test_fp16_scaler_and_single_gpu_list(checkpoint_900):
    import numpy as np
    import torch

    (checkpoint_900.parent / "run_metadata.json").write_text(json.dumps({
        "compute_dtype": "torch.float16",
    }))
    torch.save({"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                "growth_interval": 2000, "_growth_tracker": 100}, checkpoint_900 / "scaler.pt")
    torch.save({"python": random.Random(42).getstate(),
                "numpy": np.random.RandomState(42).get_state(),
                "cpu": torch.Generator().get_state(), "cuda": [torch.ones(16, dtype=torch.uint8)]},
               checkpoint_900 / "rng_state.pth")
    report = verify_checkpoint_payloads(checkpoint_900)
    assert "scaler.pt" in report["file_sha256"]
    assert report["cuda_rng_format"] == "single_tensor_list"


def test_rng_unsafe_pickle_has_no_fallback(checkpoint_900):
    import torch

    torch.save(Path("not an RNG state"), checkpoint_900 / "rng_state.pth")
    with pytest.raises(pickle.UnpicklingError):
        verify_checkpoint_payloads(checkpoint_900)


@pytest.mark.parametrize("recover", [False, True])
def test_main_recovery_is_explicit(checkpoint_900, monkeypatch, recover):
    from scripts import resume_pinned_m1

    root = checkpoint_900.parents[3]
    release = root / "release"
    (release / "scripts").mkdir(parents=True)
    pipeline = release / "scripts/run_pipeline.py"
    pipeline.write_text('''
import json
from pathlib import Path

def load_environment(path):
    return {key: Path(value) for key, value in json.loads(path.read_text()).items()}

def collect_provenance(*args, **kwargs):
    return {}

def verify_resume(checkpoint, run_root, provenance, *, recover_interrupted=False):
    if not recover_interrupted:
        raise ValueError("Recovery must be explicit")
    return checkpoint
''')
    (checkpoint_900.parent.parent / "execution.json").write_text(json.dumps({
        "provenance": {"source_files": {
            "scripts/run_pipeline.py": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
        }},
    }))
    environment = root / "environment.json"
    environment.write_text(json.dumps({
        "run_root": str(checkpoint_900.parents[2]),
        "annotations": str(root / "annotations"), "image_checksums": str(root / "checksums"),
    }))
    arguments = ["resume_pinned_m1.py", "--source", str(release),
                 "--environment", str(environment), "--checkpoint", str(checkpoint_900),
                 "--output-root", str(root / "new_runs"), "--stop-after-step", "1000",
                 "--check-only"]
    if recover:
        arguments.append("--recover-interrupted")
    monkeypatch.setattr(resume_pinned_m1.sys, "argv", arguments)
    monkeypatch.setattr(resume_pinned_m1.sys, "path", list(resume_pinned_m1.sys.path))
    if recover:
        main()
    else:
        with pytest.raises(ValueError, match="Recovery must be explicit"):
            main()
    assert not (root / "new_runs").exists()


def test_main_full_epoch_uses_final_artifact_validation(checkpoint_900, monkeypatch):
    from scripts import resume_pinned_m1

    root = checkpoint_900.parents[3]
    checkpoint = checkpoint_900.with_name("checkpoint-1200")
    shutil.copytree(checkpoint_900, checkpoint)
    (checkpoint / "trainer_state.json").write_text(json.dumps(epoch_state()))
    release = root / "epoch_release"
    (release / "scripts").mkdir(parents=True)
    pipeline = release / "scripts/run_pipeline.py"
    pipeline.write_text('''
import json
from datetime import datetime, timezone
from pathlib import Path

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def load_environment(path):
    return {key: Path(value) for key, value in json.loads(path.read_text()).items()}

def collect_provenance(*args, **kwargs):
    return {}

def verify_resume(checkpoint, run_root, provenance, *, recover_interrupted=False):
    assert not recover_interrupted
    return checkpoint

def write_metadata(path, content):
    path.write_text(json.dumps(content))

def heartbeat_loop(path, runner_id, stop):
    stop.wait()

def verify_training_artifacts(run, stop):
    assert stop is None
    return "passed"
''')
    (checkpoint.parent.parent / "execution.json").write_text(json.dumps({
        "provenance": {"source_files": {
            "scripts/run_pipeline.py": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
        }},
    }))
    environment = root / "epoch_environment.json"
    environment.write_text(json.dumps({
        "run_root": str(checkpoint.parents[2]), "annotations": str(root / "annotations"),
        "image_root": str(root / "images"), "image_checksums": str(root / "checksums"),
    }))
    commands, checks = [], []

    class Process:
        stdout = []
        pid = 123

        def __init__(self, command, **kwargs):
            commands.append(command)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    def check_payload(path):
        checks.append(path.name)
        return {"global_step": int(path.name.removeprefix("checkpoint-"))}

    monkeypatch.setattr(resume_pinned_m1, "verify_checkpoint_payloads", check_payload)
    monkeypatch.setattr(resume_pinned_m1, "verify_epoch_completion",
                        lambda run, target: {"epoch": 1.0, "final_optimizer_step": target})
    monkeypatch.setattr(resume_pinned_m1.subprocess, "Popen", Process)
    monkeypatch.setattr(resume_pinned_m1.sys, "path", list(resume_pinned_m1.sys.path))
    monkeypatch.setattr(resume_pinned_m1.sys, "argv", [
        "resume_pinned_m1.py", "--source", str(release), "--environment", str(environment),
        "--checkpoint", str(checkpoint), "--output-root", str(root / "epoch_runs"),
        "--stop-after-step", "3683", "--budget-protocol", EPOCH_BUDGET,
    ])
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0
    assert len(commands) == 1 and "--stop-after-step" not in commands[0]
    assert checks == ["checkpoint-1200", "checkpoint-3683"]
    execution = json.loads(next((root / "epoch_runs").glob("*/execution.json")).read_text())
    assert execution["status"] == "passed" and execution["returncode"] == 0
    assert execution["budget"]["completion_mode"] == "full_epoch"
    assert execution["budget"]["planned_updates"] == 2483
    assert execution["epoch_completion"]["epoch"] == 1.0


def test_resume_original_root(tmp_path):
    root = tmp_path / "runs"
    checkpoint = root / "sft-full_old/training/checkpoint-800"
    assert resume_run_root({}, checkpoint, root) == root.resolve()


@pytest.mark.parametrize("fault", [None, "missing", "wrong_output", "wrong_original"])
def test_resume_relocated_root(tmp_path, fault):
    original, relocated = tmp_path / "runs", tmp_path / "training_runs"
    checkpoint = relocated / "sft-full_old/training/checkpoint-800"
    previous = {
        "paths": {"run_root": str(original)},
        "output_relocation": {"original_run_root": str(original), "output_root": str(relocated)},
    }
    if fault == "missing":
        previous.pop("output_relocation")
    elif fault == "wrong_output":
        previous["output_relocation"]["output_root"] = str(tmp_path / "other")
    elif fault == "wrong_original":
        previous["output_relocation"]["original_run_root"] = str(tmp_path / "other")
    if fault:
        with pytest.raises(ValueError):
            resume_run_root(previous, checkpoint, original)
    else:
        assert resume_run_root(previous, checkpoint, original) == relocated.resolve()


def test_duplicate_followup_is_read_only(tmp_path):
    checkpoint = (tmp_path / "old/training/checkpoint-800").resolve()
    output = tmp_path / "new"
    run = output / "sft-full_existing"
    run.mkdir(parents=True)
    metadata = run / "execution.json"
    original = json.dumps({"resume_from_checkpoint": str(checkpoint), "status": "paused"})
    metadata.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="follow-up run already exists"):
        verify_no_followup(checkpoint, output)
    assert metadata.read_text(encoding="utf-8") == original
    verify_no_followup((tmp_path / "old/training/checkpoint-900").resolve(), output)


def test_notebook_pins_launcher_and_stop():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / M1_NOTEBOOK).read_text(encoding="utf-8")
    )
    cell = cell_source(notebook, "xGYhAgj9ltcL")
    # Preserve the historical 1200-step launcher.
    digest = "8589d917d788f5706077337ee467aef46375a1c2e4fa735980267a5b34061cd3"
    assert f"LAUNCHER_SHA = '{digest}'" in cell
    assert "'--stop-after-step', '1200'" in cell
    assert "runs/m1/sft-full_20261004_065122_901248/training/checkpoint-1100" in cell
    assert "'--budget-protocol', 'vqa_nle_budget_v2_m1_1200'" in cell
    assert "'--recover-interrupted'" in cell
    assert "OUTPUT_ROOT = ROOT / 'runs/m1'" in cell
    assert "launcher = ROOT / 'source'" in cell
    assert "command + ['--check-only']" in cell
    assert cell.index("command + ['--check-only']") < cell.index("assert torch.cuda.is_available()")
    assert cell.index("assert torch.cuda.is_available()") < cell.index("subprocess.Popen(command")
    assert "launcher.is_file()" in cell
    assert "launcher.open" not in cell and "launcher.parent.mkdir" not in cell
    compile(cell, "resume_cell", "exec")


def test_resume_cell_does_not_create_training_without_gpu(notebook_checkpoint_1100):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / M1_NOTEBOOK).read_text("utf-8"))
    source = cell_source(notebook, "xGYhAgj9ltcL")
    launcher_bytes = (root / "scripts/resume_pinned_m1.py").read_bytes()
    digest = hashlib.sha256(launcher_bytes).hexdigest()
    source = source.replace(
        "8589d917d788f5706077337ee467aef46375a1c2e4fa735980267a5b34061cd3", digest
    )
    project = notebook_checkpoint_1100.parents[4]
    launcher = project / f"source/resume_pinned_m1_{digest[:12]}.py"
    launcher.parent.mkdir()
    launcher.write_bytes(launcher_bytes)
    calls = []

    def start_training(*args, **kwargs):
        raise AssertionError("Training must not start on CPU")

    namespace = {
        "P0_PASSED": True, "ROOT": project, "REPO": root, "Path": Path,
        "json": json, "hashlib": hashlib, "sys": SimpleNamespace(executable="python"),
        "print": lambda *args: None,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: calls.append(args[0]),
                                      Popen=start_training),
    }
    original_runs = sorted((project / "runs/m1").iterdir())
    with pytest.raises(AssertionError, match="Checkpoint checked on CPU"):
        exec(source, namespace)
    assert len(calls) == 1 and calls[0][-1] == "--check-only"
    assert sorted((project / "runs/m1").iterdir()) == original_runs


def test_resume_cell_preserves_input_checkpoint(notebook_checkpoint_1100):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / M1_NOTEBOOK).read_text("utf-8"))
    source = cell_source(notebook, "xGYhAgj9ltcL")
    launcher_bytes = (root / "scripts/resume_pinned_m1.py").read_bytes()
    digest = hashlib.sha256(launcher_bytes).hexdigest()
    source = source.replace(
        "8589d917d788f5706077337ee467aef46375a1c2e4fa735980267a5b34061cd3", digest
    )
    project = notebook_checkpoint_1100.parents[4]
    launcher = project / f"source/resume_pinned_m1_{digest[:12]}.py"
    launcher.parent.mkdir()
    launcher.write_bytes(launcher_bytes)
    before = {str(file.relative_to(project)): file.read_bytes()
              for file in (project / "runs/m1").rglob("*") if file.is_file()}
    calls = []

    class FakeProcess:
        stdout = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    def start(command, **kwargs):
        calls.append(command)
        return FakeProcess()

    namespace = {
        "P0_PASSED": True, "ROOT": project, "REPO": root,
        "json": json, "hashlib": hashlib, "sys": SimpleNamespace(executable="python"),
        "print": lambda *args, **kwargs: None,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        "subprocess": SimpleNamespace(run=lambda command, **kwargs: calls.append(command),
                                      Popen=start, PIPE=-1, STDOUT=-2),
    }
    exec(source, namespace)
    assert len(calls) == 2 and calls[0] == calls[1] + ["--check-only"]
    command = calls[1]
    assert command[command.index("--output-root") + 1] == str(project / "runs/m1")
    assert command[command.index("--checkpoint") + 1] == str(notebook_checkpoint_1100)
    assert command[command.index("--stop-after-step") + 1] == "1200"
    assert command[command.index("--budget-protocol") + 1] == EXTENDED_BUDGET
    assert "--recover-interrupted" in command
    assert before == {str(file.relative_to(project)): file.read_bytes()
                      for file in (project / "runs/m1").rglob("*") if file.is_file()}


@pytest.mark.parametrize("fault", [None, "failed_run", "paused", "wrong_stage", "wrong_step",
                                  "wrong_schedule", "tampered_launcher", "no_gpu",
                                  "resume_guard", "trainer_error"])
def test_epoch_cell_checks_saved_checkpoint_and_recovery_guard_before_training(tmp_path, fault):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / M1_NOTEBOOK).read_text(encoding="utf-8")
    )
    source = cell_source(notebook, "portable-m1")
    run = tmp_path / "runs/m1/sft-full_20261006_132346_065189"
    checkpoint = run / "training/checkpoint-3300"
    checkpoint.mkdir(parents=True)
    (run / "execution.json").write_text(json.dumps({
        "stage": "other" if fault == "wrong_stage" else "sft-full",
        "status": "paused" if fault == "paused" else (
            "failed" if fault == "failed_run" else "running"),
    }))
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 3200 if fault == "wrong_step" else 3300, "log_history": [],
        "max_steps": 3683, "num_train_epochs": 2 if fault == "wrong_schedule" else 1,
    }))
    launcher_bytes = (root / "scripts/resume_pinned_m1.py").read_bytes()
    digest = hashlib.sha256(launcher_bytes).hexdigest()
    assert f"EPOCH_LAUNCHER_SHA = '{digest}'" in source
    launcher = tmp_path / f"source/resume_pinned_m1_{digest[:12]}.py"
    launcher.parent.mkdir()
    launcher.write_bytes(b"changed" if fault == "tampered_launcher" else launcher_bytes)
    before = {file: file.read_bytes() for file in run.rglob("*") if file.is_file()}
    calls = []

    class Process:
        stdout = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 1 if fault == "trainer_error" else 0

    def start(command, **kwargs):
        calls.append(command)
        return Process()

    def check(command, **kwargs):
        calls.append(command)
        if fault == "resume_guard":
            raise ValueError("Resume guard rejected the checkpoint")

    namespace = {
        "P0_PASSED": True, "ROOT": tmp_path, "REPO": root, "json": json,
        "hashlib": hashlib, "sys": SimpleNamespace(executable="python"),
        "print": lambda *args, **kwargs: None,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: fault != "no_gpu")),
        "subprocess": SimpleNamespace(run=check,
                                      Popen=start, PIPE=-1, STDOUT=-2),
    }
    if fault in {"paused", "wrong_stage", "wrong_step", "wrong_schedule",
                 "tampered_launcher", "no_gpu"}:
        with pytest.raises(AssertionError):
            exec(source, namespace)
        assert len(calls) == (1 if fault == "no_gpu" else 0)
    elif fault == "resume_guard":
        with pytest.raises(ValueError, match="Resume guard rejected"):
            exec(source, namespace)
        assert len(calls) == 1 and calls[0][-1] == "--check-only"
    elif fault == "trainer_error":
        with pytest.raises(RuntimeError, match="continuation stopped"):
            exec(source, namespace)
    else:
        exec(source, namespace)
        assert calls[0] == calls[1] + ["--check-only"]
        assert calls[1][calls[1].index("--stop-after-step") + 1] == "3683"
        assert calls[1][calls[1].index("--budget-protocol") + 1] == EPOCH_BUDGET
        assert ("--recover-interrupted" in calls[1]) == (fault != "failed_run")
    assert before == {file: file.read_bytes() for file in run.rglob("*") if file.is_file()}


@pytest.mark.parametrize("missing", [None, "token_full_M1_v1.json", "m1_environment_v1.json"])
def test_saved_preflight_is_reused_without_rescanning(tmp_path, missing):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / M1_NOTEBOOK).read_text(encoding="utf-8")
    )
    source = cell_source(notebook, "portable-token-full")
    source = source.split("RUN_ROOT.mkdir", 1)[0]
    recorded = tmp_path / "recorded_runs"
    recorded.mkdir()
    for name in ("token_full_M1_v1.json", "m1_environment_v1.json"):
        if name != missing:
            (recorded / name).write_text("{}")
    execution = tmp_path / "training_runs/sft-full_20261002_113657_805762/execution.json"
    execution.parent.mkdir(parents=True)
    execution.write_text(json.dumps({"paths": {"run_root": str(recorded)}}))
    namespace = {"ROOT": tmp_path, "Path": Path, "json": json}
    if missing:
        with pytest.raises(FileNotFoundError, match="do not rescan"):
            exec(source, namespace)
    else:
        exec(source, namespace)
        assert namespace["RUN_ROOT"] == recorded
        assert not (tmp_path / "runs").exists()
