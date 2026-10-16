import ast
import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest


def _source_verifiers():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/08_m1_optimizer_pilot.ipynb").read_text(encoding="utf-8")
    )
    source_cell = next(cell for cell in notebook["cells"] if cell["id"] == "m1-pilot-source")
    source = "".join(source_cell["source"])
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"ARCHIVE_ROOT", "MANIFEST_MEMBER"}
                for target in node.targets
            )
        ):
            selected.append(node)
    namespace = {"hashlib": hashlib, "json": json, "stat": stat}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "source-cell", "exec"), namespace)
    return namespace


def test_m1_pilot_notebook_verifies_inputs_and_never_starts_full_training():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/archive/08_m1_optimizer_pilot.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 6
    assert [cell["id"] for cell in cells[:4]] == [
        "m1-pilot-mount",
        "m1-pilot-source",
        "m1-pilot-data",
        "m1-resume-rehearsal",
    ]
    source = "\n".join("".join(cell["source"]) for cell in cells)
    for cell in cells:
        compile("".join(cell["source"]), cell["id"], "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    assert source.count("drive.mount") == 1
    assert "torch.cuda.is_available()" in source
    source_setup = "".join(cells[1]["source"])
    assert "SOURCE_RELEASE = 'source_20260924_072646_069371'" in source_setup
    assert (
        "EXPECTED_SOURCE_SHA256 = "
        "'a797a32fe2dc5c36196aa7c68ec0bc9b4f52ec222faaa5313c4b49db22d6b333'"
    ) in source_setup
    assert "checksum_parts != [EXPECTED_SOURCE_SHA256, SOURCE_ZIP.name]" in source_setup
    assert "digest.hexdigest() != EXPECTED_SOURCE_SHA256" in source_setup
    assert "glob('source_*.sha256')" not in source_setup
    assert "_verify_source_archive(archive)" in source_setup
    assert "TemporaryDirectory(" in source_setup
    assert "_verify_source_tree(REPO, manifest, manifest_hash)" in source_setup
    assert "f'{REPO}[train]'" in source
    assert "verify_coco_bundle.py" in source
    assert "rehearse_m1_resume.py" in "".join(cells[3]["source"])
    assert "max_lora_b_weight_change" in "".join(cells[3]["source"])
    assert "sft-full-preflight" in source
    assert "sft-optimizer-pilot" in source
    assert "'sft-full'," not in source
    assert "max_lora_weight_change" in source
    assert "trainer_state.json" in source
    assert "Qwen25VLRunner" in source
    assert "adapter_reload.json" in source
    for cell, function_name in zip(
        cells[-2:],
        ("run_standalone_pilot", "reload_standalone_pilot_adapter"),
        strict=True,
    ):
        statements = ast.parse("".join(cell["source"])).body
        assert len(statements) == 1
        assert isinstance(statements[0], ast.FunctionDef)
        assert statements[0].name == function_name


def _source_zip(path, relative="scripts/run_pipeline.py", content=b"pass\n", extra=None):
    namespace = _source_verifiers()
    root = namespace["ARCHIVE_ROOT"]
    manifest = {relative: hashlib.sha256(content).hexdigest()}
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr(root + relative, content)
        archive.writestr(root + "SOURCE_MANIFEST.json", json.dumps(manifest))
        if extra is not None:
            archive.writestr(root + extra, b"pass\n")
    return manifest


def test_source_cell_rejects_unmanifested_or_unsafe_zip_members(tmp_path):
    namespace = _source_verifiers()
    valid_zip = tmp_path / "valid.zip"
    _source_zip(valid_zip)
    with zipfile.ZipFile(valid_zip) as archive:
        manifest, manifest_hash = namespace["_verify_source_archive"](archive)
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)
    repo = extracted / "Vietnamese-VQA-Explanation"
    namespace["_verify_source_tree"](repo, manifest, manifest_hash)
    (repo / "scripts/run_pipeline.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        namespace["_verify_source_tree"](repo, manifest, manifest_hash)
    (repo / "scripts/run_pipeline.py").write_bytes(b"pass\n")
    (repo / "scripts/extra.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected executable"):
        namespace["_verify_source_tree"](repo, manifest, manifest_hash)

    extra_zip = tmp_path / "extra.zip"
    _source_zip(extra_zip, extra="scripts/extra.py")
    with zipfile.ZipFile(extra_zip) as archive:
        with pytest.raises(ValueError, match="members"):
            namespace["_verify_source_archive"](archive)

    unsafe_zip = tmp_path / "unsafe.zip"
    _source_zip(unsafe_zip, relative="../escape.py")
    with zipfile.ZipFile(unsafe_zip) as archive:
        with pytest.raises(ValueError, match="Unsafe source path"):
            namespace["_verify_source_archive"](archive)
