import json
from pathlib import Path


def test_batch_pilot_notebook_has_gpu_guard_and_no_full_train():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/07_sft_batch_pilot.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 4
    source = "\n".join("".join(cell["source"]) for cell in cells)
    for cell in cells:
        compile("".join(cell["source"]), cell["id"], "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    assert source.count("drive.mount") == 1
    assert "torch.cuda.is_available()" in source
    assert "verify_coco_bundle.py" in source
    assert "sft-batch-pilot" in source
    assert "EXPECTED_BUNDLE_SHA256" in source
    assert "batch_pilot.json" in source
    assert "train_sft.py" not in source
