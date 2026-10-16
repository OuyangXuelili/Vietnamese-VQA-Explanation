"""Preserve the historical smoke notebook while the canonical source evolves."""

import ast
import hashlib
import json
import re
from pathlib import Path


def test_legacy_notebook_compiles_and_is_frozen():
    root = Path(__file__).resolve().parents[2]
    text = (root / "notebooks/archive/02_sft_smoke.ipynb").read_text(encoding="utf-8")
    notebook = json.loads(text)
    overlay = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(tree, "notebook", "exec")
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "OVERLAY" for t in node.targets
            ):
                source_json = ast.literal_eval(node.value.args[0])
                # Freeze the historical training source while allowing status/verification cells.
                assert hashlib.sha256(source_json.encode()).hexdigest() == (
                    "9876ffb3104dc788627e2289e60a8a4493b514ce243891b12be2390e81416f73"
                )
                overlay = json.loads(source_json)
    assert overlay
    for relative, content in overlay.items():
        if relative.endswith(".py"):
            compile(content, relative, "exec")
    assert notebook["cells"][-2]["id"] == "EtQqRkD-8EWg"
    assert notebook["cells"][-1]["cell_type"] == "markdown"
    assert "Kết quả chạy thử" in "".join(notebook["cells"][-1]["source"])
    assert all(notebook["cells"][i]["outputs"] for i in range(1, 6))


def test_archived_run_outputs_show_completed_smoke():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/02_sft_smoke.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 5
    assert all(
        output["output_type"] != "error" for cell in code_cells for output in cell["outputs"]
    )

    def stdout(cell):
        return "".join("".join(output.get("text", [])) for output in cell["outputs"])

    run_id = "M1_smoke_20260922_065719"
    assert "Tesla T4" in stdout(code_cells[0])
    assert f"READY: /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/{run_id}" in stdout(
        code_cells[1]
    )
    result = json.loads(stdout(code_cells[2]))
    assert result["status"] == "passed"
    assert result["global_step"] == 8
    assert result["max_lora_weight_change"] > 0
    assert "ADAPTER RELOAD + INFERENCE PASSED:" in stdout(code_cells[3])
    assert run_id in stdout(code_cells[3])

    verification = stdout(code_cells[4])
    match = re.search(r'\{\s*"n":.*?\n\}', verification, flags=re.DOTALL)
    assert match is not None
    metrics = json.loads(match.group())
    assert metrics["n"] == metrics["format_valid_n"] == 20
    assert metrics["format_rate"] == 1.0
    assert "VERIFIED: 20 unique predictions; adapter files present" in verification
    assert f"RUN: /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/{run_id}" in verification


def test_historical_notebook_has_one_mount_and_shared_imports():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/02_sft_smoke.ipynb").read_text(encoding="utf-8")
    )
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert sum(source.count("drive.mount(") for source in sources) == 1
    assert sum(source.count("import json\n") for source in sources) == 1
    assert sum(source.count("from pathlib import Path\n") for source in sources) == 1
    assert "torch.cuda.is_available()" in sources[1]
    assert "Chạy ô khởi tạo" in sources[-1]
