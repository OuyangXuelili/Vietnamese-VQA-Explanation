"""Environment setup does not install into the notebook's interpreter."""

import subprocess

import pytest

from vqa_explanation.utils import cloud_environment as cloud


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "requirements-cloud-preference.txt").write_text("-e .\n", encoding="utf-8")
    return root


def create_fake_environment(directory):
    directory.mkdir(parents=True)
    (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    python = cloud.environment_python(directory)
    python.parent.mkdir()
    python.touch()


def test_new_environment_is_isolated_and_checked(project, tmp_path, monkeypatch):
    calls = []

    class Builder:
        def __init__(self, **kwargs):
            assert kwargs == {"with_pip": False, "system_site_packages": False}

        def create(self, directory):
            create_fake_environment(directory)

    monkeypatch.setattr(cloud.venv, "EnvBuilder", Builder)
    directory = tmp_path / "training_env"
    python = cloud.prepare_environment(project, directory,
                                       runner=lambda command, **kwargs: calls.append(command))
    assert python == cloud.environment_python(directory)
    assert all(command[3:5] == ["--python", str(python)] for command in calls)
    assert "install" in calls[0] and calls[1][-1] == "check"
    assert str(project / "requirements-cloud-preference.txt") in calls[0]


def test_valid_environment_is_reused_without_recreating_it(project, tmp_path, monkeypatch):
    directory = tmp_path / "training_env"
    create_fake_environment(directory)
    monkeypatch.setattr(cloud.venv, "EnvBuilder", lambda **kwargs: pytest.fail("Must not recreate"))
    calls = []
    cloud.prepare_environment(project, directory,
                              runner=lambda command, **kwargs: calls.append(command))
    assert len(calls) == 2


def test_unknown_directory_is_not_modified(project, tmp_path):
    directory = tmp_path / "other"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="not a virtual environment"):
        cloud.prepare_environment(project, directory, runner=lambda *args, **kwargs: pytest.fail())
    assert marker.read_text() == "keep" and list(directory.iterdir()) == [marker]


def test_source_ancestor_cannot_be_an_environment(project):
    with pytest.raises(ValueError, match="ancestors"):
        cloud.prepare_environment(project, project.parent)


def test_environment_with_system_packages_is_rejected(project, tmp_path):
    directory = tmp_path / "training_env"
    create_fake_environment(directory)
    (directory / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="exclude system"):
        cloud.prepare_environment(project, directory, runner=lambda *args, **kwargs: pytest.fail())


def test_dependency_failure_is_not_reported_as_success(project, tmp_path):
    directory = tmp_path / "training_env"
    create_fake_environment(directory)

    def runner(command, **kwargs):
        if command[-1] == "check":
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        cloud.prepare_environment(project, directory, runner=runner)
