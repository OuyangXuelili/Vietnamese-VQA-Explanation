import ast
import json

from vqa_explanation.utils.environment import project_root


def test_full_data_notebook_is_clean_and_uses_verified_release():
    path = project_root() / "notebooks/archive/05_full_data_preflight.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        ast.parse(source)
        code.append(source)
    source = "\n".join(code)
    assert "drive.mount" in source
    assert "sha256" in source
    assert "verify_coco_bundle.py" in source
    assert "46846a81c7b3594d4fc9670ec51019cfd652f1eb169af7978e39dc5a40a55522" in source
    assert "--extract-to" in source
    assert "--all-train-records" in source
    assert 'for stage in ("audit", "preflight")' in source
    assert "sft-smoke" not in source
