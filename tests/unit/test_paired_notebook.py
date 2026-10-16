"""Check the inference-only notebook orchestration without claiming a GPU test."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_paired_notebook_compiles_and_preserves_scope():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/archive/04_paired_smoke.ipynb").read_text("utf-8"))
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    for name, source in cells.items():
        compile(source, name, "exec")
    assert "train_sft" not in "".join(cells.values())
    # Both commands must be derived from the same shared CLI argument list.
    tree = ast.parse(cells["paired"])
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_cmd"
    ]
    assert len(calls) == 2
    for call in calls:
        assert isinstance(call.args[1], ast.BinOp)
        assert isinstance(call.args[1].left, ast.Name)
        assert call.args[1].left.id == "common"
    comparison = ast.unparse(ast.parse(cells["compare"]))
    assert "a[qid]['image_sha256'] == b[qid]['image_sha256']" in comparison
    assert "a[qid]['compute_dtype'] == b[qid]['compute_dtype']" in comparison
    assert "len(a) == len(b) == 20" in cells["compare"]


def test_paired_setup_streams_hashes_and_keeps_freeze_output_in_log():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/archive/04_paired_smoke.ipynb").read_text("utf-8"))
    setup = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "setup")
    assert setup.count("drive.mount(") == 1
    assert 'for chunk in iter(lambda: source.read(1024 * 1024), b"")' in setup
    assert "SOURCE.read_bytes()" not in setup
    assert "ARCHIVE.read_bytes()" not in setup
    assert 'adapter_model.safetensors").read_bytes()' not in setup
    assert 'run_cmd("environment", [sys.executable, "-m", "pip", "freeze"], echo=False)' in setup
    assert "log.write(line)" in setup


def test_paired_setup_helpers_work_without_colab(tmp_path, capsys):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/archive/04_paired_smoke.ipynb").read_text("utf-8"))
    setup = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "setup")
    functions = [
        node
        for node in ast.parse(setup).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"sha256_file", "write_json", "run_cmd"}
    ]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "subprocess": subprocess,
        "REPO": tmp_path,
        "RUN": tmp_path,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "setup-helpers", "exec"), namespace)
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"abc" * 400_000)
    assert namespace["sha256_file"](sample) == hashlib.sha256(sample.read_bytes()).hexdigest()
    namespace["run_cmd"]("quiet", [sys.executable, "-c", "print('saved only')"], echo=False)
    assert capsys.readouterr().out == ""
    assert (tmp_path / "quiet.log").read_text(encoding="utf-8").strip() == "saved only"
    assert json.loads((tmp_path / "quiet_command.json").read_text(encoding="utf-8"))[-1] == (
        "print('saved only')"
    )
