import subprocess
import sys
from pathlib import Path

import pytest


def test_ci_option_rejects_a_skipped_test(tmp_path):
    configuration = Path(__file__).resolve().parents[1] / "conftest.py"
    (tmp_path / "conftest.py").write_bytes(configuration.read_bytes())
    (tmp_path / "test_optional.py").write_text(
        "import pytest\ndef test_optional(): pytest.skip('missing dependency')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "--fail-on-skip", "-q"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("skip_current", [False, True])
def test_source_only_ci_excludes_history_but_rejects_missing_current_checks(
    tmp_path, skip_current,
):
    configuration = Path(__file__).resolve().parents[1] / "conftest.py"
    (tmp_path / "conftest.py").write_bytes(configuration.read_bytes())
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = historical: requires saved runs\n", encoding="utf-8",
    )
    current = "pytest.skip('missing required dependency')" if skip_current else "assert True"
    (tmp_path / "test_checks.py").write_text(
        "import pytest\n"
        "@pytest.mark.historical\ndef test_history(): pytest.skip('artifact not shipped')\n"
        f"def test_current(): {current}\n", encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-m", "not historical",
         "--fail-on-skip", "-q"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == (1 if skip_current else 0), result.stdout + result.stderr
    assert "1 deselected" in result.stdout
