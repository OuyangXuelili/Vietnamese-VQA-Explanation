import json
from pathlib import Path


def test_token_profile_notebook_is_cpu_only_and_has_valid_cells():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/06_sft_token_profile.ipynb").read_text(encoding="utf-8")
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
    assert "verify_coco_bundle.py" in source
    assert "token-profile" in source
    assert "EXPECTED_BUNDLE_SHA256" in source
    assert "train_sft.py" not in source
