"""Task receipts around the existing checks; no model or account orchestration."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from filelock import FileLock

from .process import run_command
from .run_provenance import git_state, sha256_file, utc_now

KINDS = ("code", "data", "frontend", "report", "docs")
PROFILES = ("focused", "full", "data")
STATES = ("in_progress", "blocked", "needs_input", "completed")
FOLDERS = ("src", "scripts", "tests", "configs", "app", "docs", ".github",
           "notebooks", "reports/thesis", "reports/presentation")
ROOT_FILES = ("AGENTS.md", "TIEN_DO.md", "README.md", "pyproject.toml", "Makefile",
              ".gitignore", ".gitattributes", ".env.example", "uv.lock")
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".html", ".css",
            ".js", ".mjs", ".json", ".csv", ".ipynb", ".docx", ".pdf", ".pptx",
            ".png", ".svg"}
IGNORED = {"__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".ipynb_checkpoints"}
ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")
    return value.strip()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Path must stay inside the repository")
    relative = candidate.absolute().relative_to(root)
    if any(root.joinpath(*relative.parts[:i]).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Linked paths are not accepted")
    return resolved


def _file_record(root: Path, path: Path) -> dict:
    path = _inside(root, path)
    if not path.is_file():
        raise ValueError(f"Evidence file missing: {path}")
    relative = path.relative_to(root)
    if any(part.lower() in {".git", ".venv", ".env", "auth", "credentials"}
           for part in relative.parts):
        raise ValueError("Credential and environment files are not evidence")
    stat = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while reading: {relative}")
    return {"path": relative.as_posix(), "sha256": digest, "bytes": stat.st_size}


def worktree_files(root: Path) -> dict[str, str]:
    """Snapshot working source, tests and documents, including pre-existing edits."""
    root = root.resolve()
    files = {root / name for name in ROOT_FILES if (root / name).is_file()}
    files.update(root.glob("requirements*.txt"))
    for folder in FOLDERS:
        base = root / folder
        if base.is_symlink():
            raise ValueError("Linked source directory")
        if not base.is_dir():
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            parent = Path(directory)
            dirs[:] = [name for name in dirs if name not in IGNORED
                       and (parent / name).relative_to(root).as_posix() != "notebooks/archive"]
            if any((parent / name).is_symlink() for name in dirs):
                raise ValueError("Linked source directory")
            files.update(parent / name for name in names if Path(name).suffix in SUFFIXES)
    return {item["path"]: item["sha256"] for item in
            (_file_record(root, path) for path in sorted(files))}


def _verification_path(name: str) -> bool:
    return (name.startswith(("src/", "scripts/", "tests/", "configs/", "app/", ".github/",
                             "notebooks/"))
            or name in {"pyproject.toml", "Makefile", ".env.example", "uv.lock"}
            or name.startswith("requirements"))


def verification_files(root: Path) -> dict[str, str]:
    """Inputs affecting code checks; journal/report edits do not invalidate code tests."""
    return {name: digest for name, digest in worktree_files(root).items()
            if _verification_path(name)}


def verification_inputs(root: Path) -> dict:
    packages = {}
    for name in ("ruff", "pytest", "torch", "torchvision", "transformers", "accelerate",
                 "peft", "fastapi", "uvicorn", "httpx", "PyYAML", "Pillow", "numpy",
                 "bert-score", "matplotlib", "filelock"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    node = shutil.which("node")
    node_stat = Path(node).stat() if node else None
    return {"files": verification_files(root), "python": sys.version,
            "executable": sys.executable, "packages": packages,
            "node": {"path": node, "bytes": node_stat.st_size if node_stat else None,
                     "mtime_ns": node_stat.st_mtime_ns if node_stat else None}}


def _directory(root: Path, task: Path) -> Path:
    path = _inside(root, task)
    base = root / "artifacts/tasks"
    if path.parent != base or not ID_PATTERN.fullmatch(path.name):
        raise ValueError("Task must be a receipt directory under artifacts/tasks")
    return path


def _load(root: Path, task: Path) -> tuple[Path, dict]:
    path = _directory(root, task)
    record = json.loads((path / "task.json").read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != "vqa_task_v1"
            or record.get("id") != path.name or record.get("repository") != str(root)
            or record.get("kind") not in KINDS or record.get("status") not in STATES
            or not isinstance(record.get("criteria"), list) or not record["criteria"]
            or not all(isinstance(item, str) and item.strip() for item in record["criteria"])
            or not isinstance(record.get("checks"), list)
            or not all(isinstance(item, str) and ID_PATTERN.fullmatch(item)
                       for item in record["checks"])
            or not isinstance(record.get("baseline"), dict)
            or not all(isinstance(name, str) and isinstance(value, str)
                       for name, value in record["baseline"].items())
            or not isinstance(record.get("handoffs"), list)
            or not isinstance(record.get("reviews"), list)
            or not all(isinstance(item, dict) and type(item.get("criterion")) is int
                       and item.get("result") in {"passed", "failed", "blocked"}
                       and isinstance(item.get("verification_inputs"), dict)
                       and isinstance(item.get("evidence"), list)
                       for item in record["reviews"])):
        raise ValueError("Invalid task receipt or wrong repository")
    return path, record


def _mutable(record: dict) -> None:
    if record["status"] == "completed":
        raise ValueError("Completed tasks are closed; start a new task for further work")


def start(root: Path, title: str, kind: str, criteria: list[str]) -> Path:
    root = root.resolve()
    if kind not in KINDS or not criteria:
        raise ValueError("Choose a task kind and at least one acceptance criterion")
    criteria = [_text(item, "Acceptance criterion") for item in criteria]
    title = _text(title, "Title")
    baseline = worktree_files(root)
    path = _directory(root, Path("artifacts/tasks") / uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    _write(path / "task.json", {
        "schema": "vqa_task_v1", "id": path.name, "repository": str(root),
        "title": title, "kind": kind, "criteria": criteria, "created_at_utc": utc_now(),
        "status": "in_progress", "baseline": baseline, "git_at_start": git_state(root),
        "context": ["AGENTS.md", "TIEN_DO.md", "docs/research_plan.md", "docs/harness.md"],
        "checks": [], "reviews": [], "handoffs": [],
    })
    return path


def check_commands(root: Path, profile: str, tests: list[str],
                   source_only: bool) -> list[list[str]]:
    if profile not in PROFILES:
        raise ValueError("Unknown check profile")
    python = [sys.executable, "-B", "-X", "utf8"]
    probe = [*python, "-c", "import pathlib,sys,vqa_explanation; "
             "p=pathlib.Path(vqa_explanation.__file__).resolve(); "
             "ok=p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()); "
             "sys.exit(0 if ok else 'Python imports another checkout: '+str(p))", str(root)]
    if profile != "focused":
        if tests or (profile == "data" and source_only):
            raise ValueError("Tests are for focused checks; full-data requires full history scope")
        command = [*python, str(root / "scripts/check_project.py")]
        if profile == "data":
            command.append("--full-data")
        if source_only:
            command.append("--source-only")
        return [probe, command]
    if not tests or source_only:
        raise ValueError("Focused checks require explicit test files; use full for source-only")
    targets = []
    for target in tests:
        file, separator, node = target.partition("::")
        path = _inside(root, Path(file))
        if not path.is_file() or path.suffix != ".py" or not path.is_relative_to(root / "tests"):
            raise ValueError("Focused targets must be existing Python tests under tests/")
        targets.append(path.relative_to(root).as_posix() + (separator + node if separator else ""))
    return [probe, [*python, "-m", "ruff", "check", "."],
            [*python, "-m", "pytest", *targets, "--fail-on-skip"]]


def check(root: Path, task: Path, profile: str = "full", tests: list[str] | None = None,
          source_only: bool = False) -> dict:
    root = root.resolve()
    path = _directory(root, task)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        if os.environ.get(name):
            raise ValueError(f"Clear {name}; declare test selection through the check profile")
    if os.environ.get("PYTHONOPTIMIZE", "0") not in {"", "0"}:
        raise ValueError("Clear PYTHONOPTIMIZE; verification must retain Python checks")
    commands = check_commands(root, profile, tests or [], source_only)
    with FileLock(str(path / "task.lock"), timeout=0):
        path, record = _load(root, path)
        _mutable(record)
        identifier = uuid.uuid4().hex
        directory = path / "checks" / identifier
        directory.mkdir(parents=True, exist_ok=False)
        receipt_path = directory / "receipt.json"
        receipt = {"schema": "vqa_task_check_v1", "id": identifier, "profile": profile,
                   "source_only": source_only, "requested_tests": tests or [],
                   "started_at_utc": utc_now(),
                   "python": sys.executable, "status": "running", "commands": [],
                   "before": verification_inputs(root), "returncode": None}
        _write(receipt_path, receipt)
        record["checks"].append(identifier)
        record["status"] = "in_progress"
        _write(path / "task.json", record)
        try:
            with (directory / "console.log").open("x", encoding="utf-8") as log:
                for command in commands:
                    step = {"argv": command, "started_at_utc": utc_now(), "returncode": None}
                    receipt["commands"].append(step)
                    _write(receipt_path, receipt)
                    try:
                        run_command(command, cwd=root, on_output=lambda line: (log.write(line),
                                                                            log.flush()))
                    except subprocess.CalledProcessError as error:
                        step["returncode"] = error.returncode
                        raise
                    else:
                        step["returncode"] = 0
                    finally:
                        step["finished_at_utc"] = utc_now()
            receipt.update(status="passed", returncode=0)
        except KeyboardInterrupt:
            receipt.update(status="interrupted", returncode=130)
        except Exception as error:
            receipt.update(status="failed", returncode=getattr(error, "returncode", 1),
                           error=f"{type(error).__name__}: {error}")
        finally:
            try:
                receipt["after"] = verification_inputs(root)
                if receipt["status"] == "passed" and receipt["after"] != receipt["before"]:
                    receipt.update(status="stale", returncode=1,
                                   error="Verification inputs changed during the checks")
            except Exception as error:
                receipt.update(status="failed", returncode=1, error=str(error))
            receipt["finished_at_utc"] = utc_now()
            if (directory / "console.log").is_file():
                receipt["log_sha256"] = sha256_file(directory / "console.log")
            _write(receipt_path, receipt)
        return {"status": receipt["status"], "returncode": receipt["returncode"],
                "receipt": str(receipt_path), "profile": profile, "source_only": source_only}


def review(root: Path, task: Path, criterion: int, evidence: list[Path], note: str,
           result: str = "passed") -> dict:
    """Record a review statement with hashed files, not an independent human label."""
    root = root.resolve()
    path = _directory(root, task)
    if result not in {"passed", "failed", "blocked"} or not evidence:
        raise ValueError("Review needs a result and evidence files")
    note = _text(note, "Review note")
    with FileLock(str(path / "task.lock"), timeout=0):
        path, record = _load(root, path)
        _mutable(record)
        if type(criterion) is not int or not 1 <= criterion <= len(record["criteria"]):
            raise ValueError("Unknown acceptance criterion")
        item = {"criterion": criterion, "result": result, "note": note,
                "method": "recorded_attestation", "independent_human_review_verified": False,
                "at_utc": utc_now(), "verification_inputs": verification_inputs(root),
                "evidence": [_file_record(root, file) for file in evidence]}
        record["reviews"].append(item)
        _write(path / "task.json", record)
        return {"criterion": criterion, "result": result, "method": item["method"]}


def _changes(before: dict, after: dict) -> list[str]:
    return sorted(name for name in before.keys() | after.keys()
                  if before.get(name) != after.get(name))


def _latest_check(root: Path, path: Path, record: dict, current: dict) -> dict | None:
    if not record["checks"]:
        return None
    receipt_path = path / "checks" / record["checks"][-1] / "receipt.json"
    receipt = json.loads(_inside(root, receipt_path).read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict) or receipt.get("schema") != "vqa_task_check_v1"
            or receipt.get("profile") not in PROFILES or receipt.get("status") != "passed"
            or receipt.get("returncode") != 0 or receipt.get("before") != current
            or receipt.get("after") != current or not receipt.get("commands")
            or any(item.get("returncode") != 0 for item in receipt["commands"])):
        raise ValueError("Latest check failed, was interrupted, or is stale; run it again")
    expected = check_commands(root, receipt["profile"], receipt["requested_tests"],
                              receipt["source_only"])
    if [step["argv"] for step in receipt["commands"]] != expected:
        raise ValueError("Recorded commands do not match the declared check profile")
    log = _inside(root, receipt_path.with_name("console.log"))
    if not log.is_file() or sha256_file(log) != receipt.get("log_sha256"):
        raise ValueError("Check log is missing or changed")
    return receipt


def finish(root: Path, task: Path, status: str, summary: str, next_step: str = "") -> Path:
    root = root.resolve()
    path = _directory(root, task)
    if status not in STATES:
        raise ValueError("Unknown handoff status")
    summary = _text(summary, "Summary")
    if status != "completed":
        next_step = _text(next_step, "Next step for unfinished work")
    with FileLock(str(path / "task.lock"), timeout=0):
        path, record = _load(root, path)
        _mutable(record)
        current = verification_inputs(root)
        changed = _changes(record["baseline"], worktree_files(root))
        if status == "completed":
            needs_check = record["kind"] in {"code", "data", "frontend"} or any(
                _verification_path(name) for name in changed)
            latest = _latest_check(root, path, record, current) if record["checks"] else None
            if needs_check and latest is None:
                raise ValueError("Code/data/frontend completion needs an executed check")
            if record["kind"] == "data" and (latest or {}).get("profile") != "data":
                raise ValueError("Data tasks require the full-data check profile")
            for index in range(1, len(record["criteria"]) + 1):
                reviews = [item for item in record["reviews"] if item["criterion"] == index]
                item = reviews[-1] if reviews else None
                if (not item or item["result"] != "passed"
                        or item["verification_inputs"] != current or not item["evidence"]):
                    raise ValueError(f"Acceptance criterion {index} is pending, failed or stale")
                for file in item["evidence"]:
                    if _file_record(root, Path(file["path"])) != file:
                        raise ValueError(f"Review evidence changed: {file['path']}")
        identifier = uuid.uuid4().hex
        directory = path / "handoffs"
        directory.mkdir(exist_ok=True)
        handoff = {"schema": "vqa_task_handoff_v1", "at_utc": utc_now(), "status": status,
                   "summary": summary, "next_step": next_step, "changed_paths": changed,
                   "checks": record["checks"], "criteria": record["criteria"],
                   "reviews": record["reviews"], "verification_inputs": current,
                   "git": git_state(root), "limitations":
                   "Recorded criteria are attestations, not independent human evaluation. "
                   "Code checks do not prove model quality, GPU execution or Drive sync."}
        output = directory / (identifier + ".json")
        _write(output, handoff)
        record["handoffs"].append(output.relative_to(path).as_posix())
        record["status"] = status
        _write(path / "task.json", record)
        return output


def inspect(root: Path, task: Path | None = None) -> dict:
    root = root.resolve()
    context = {name: {"exists": (root / name).is_file()} for name in
               ("AGENTS.md", "TIEN_DO.md", "docs/research_plan.md", "docs/harness.md")}
    if task is None:
        paths = sorted((root / "artifacts/tasks").glob("*/task.json"),
                       key=lambda path: path.stat().st_mtime_ns, reverse=True)
        tasks = []
        for file in paths[:10]:
            try:
                path, record = _load(root, file.parent)
                tasks.append({"task": str(path), "title": record["title"],
                              "status": record["status"], "handoffs": record["handoffs"][-1:]})
            except (ValueError, OSError, KeyError):
                tasks.append({"task": str(file.parent), "status": "unreadable"})
        return {"repository": str(root), "context": context, "git": git_state(root),
                "tasks": tasks, "note": "Read context and unfinished handoffs before editing."}
    path, record = _load(root, task)
    current = verification_inputs(root)
    try:
        latest = _latest_check(root, path, record, current)
        check_state = "passed" if latest else "not_run"
    except (ValueError, OSError):
        check_state = "failed_or_stale"
    return {"task": str(path), "title": record["title"], "kind": record["kind"],
            "status": record["status"], "criteria": record["criteria"],
            "latest_check": check_state, "checks": record["checks"],
            "reviews": record["reviews"], "handoffs": record["handoffs"],
            "changed_paths": _changes(record["baseline"], worktree_files(root))}
