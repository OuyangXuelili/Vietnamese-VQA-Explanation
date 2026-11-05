"""Resume a partial test export after its Drive storage paths change."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from filelock import FileLock

from vqa_explanation.evaluation.test_report import inspect_test_outputs
from vqa_explanation.inference.export import read_jsonl
from vqa_explanation.inference.export_relocation import prepare_relocation, write_relocation
from vqa_explanation.inference.study_test import (
    IDENTITIES,
    build_test_run_record,
    checked_protocol,
    prepare_test,
)
from vqa_explanation.utils.environment import load_environment
from vqa_explanation.utils.run_provenance import sha256_file


def source_hashes(directory, identity):
    names = ("test_run.json", f"{identity}.jsonl", f"{identity}.jsonl.manifest.json")
    return {name: sha256_file(directory / name) for name in names}


def verify_snapshot_prefixes(directory, identity):
    output = directory / f"{identity}.jsonl"
    rows = read_jsonl(output)
    manifest = output.with_suffix(".jsonl.manifest.json").read_bytes()
    original_output = None
    for marker in sorted((directory / "recovery").glob(f"{identity}_*/snapshot.json")):
        receipt = json.loads(marker.read_text("utf-8"))
        snapshot = marker.parent / output.name
        count = receipt.get("predictions")
        declared = receipt.get("output", "")
        declared_output = (PureWindowsPath(declared) if "\\" in declared
                           else PurePosixPath(declared))
        declared_snapshot = declared_output.parent / "recovery" / marker.parent.name / output.name
        if (receipt.get("schema") != "closed_export_snapshot_v1"
                or type(count) is not int or count < 0
                or marker.parent.name != f"{identity}_{count:04d}"
                or declared_output.name != output.name
                or receipt.get("snapshot") != str(declared_snapshot)
                or receipt.get("sha256") != sha256_file(snapshot)
                or receipt.get("manifest_sha256") != sha256_file(
                    snapshot.with_suffix(".jsonl.manifest.json"))
                or snapshot.with_suffix(".jsonl.manifest.json").read_bytes() != manifest
                or read_jsonl(snapshot) != rows[:count] or count > len(rows)):
            raise ValueError("Closed snapshot differs from the partial test prefix")
        if original_output is not None and declared_output != original_output:
            raise ValueError("Closed snapshots refer to different original exports")
        original_output = declared_output


def relocate_test_run(environment, plan_path, plan_sha256, identity, source_dir, output_dir,
                      *, adapter=None, rubric=None, inspect_only=False):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    env = load_environment(Path(environment))
    runs = {name: source_dir.parent / name for name in IDENTITIES}
    runs[identity] = source_dir
    plan, _, _, progress, _ = inspect_test_outputs(
        plan_path, plan_sha256, env["annotations"].parent / "test.jsonl", runs)
    if progress[identity]["status"] != "incomplete" or not progress[identity]["saved"]:
        raise ValueError("Relocation requires a verified, nonempty partial test export")
    verify_snapshot_prefixes(source_dir, identity)
    _, args, integrity, blockers = prepare_test(
        environment, plan_path, plan_sha256, identity, output_dir, adapter=adapter,
        rubric=rubric)
    if blockers:
        raise ValueError(" ".join(blockers))
    original = json.loads((source_dir / "test_run.json").read_text("utf-8"))
    record = build_test_run_record(args, plan, integrity)
    if {k: v for k, v in original.items() if k != "adapter"} != {
            k: v for k, v in record.items() if k != "adapter"}:
        raise ValueError("Test run differs by more than its adapter storage path")
    before = source_hashes(source_dir, identity)
    marker = output_dir / "storage_relocation_verified.json"
    if output_dir.exists():
        if not marker.is_file():
            raise ValueError("Destination exists without a verified relocation receipt")
        receipt = json.loads(marker.read_text("utf-8"))
        if (receipt.get("schema") != "test_storage_relocation_v1"
                or receipt.get("identity") != identity
                or receipt.get("plan_sha256") != plan_sha256
                or receipt.get("source_files") != before
                or receipt.get("destination") != str(output_dir)
                or sha256_file(output_dir / "test_run.json") != receipt["run_sha256"]
                or sha256_file(args.output.with_suffix(".jsonl.manifest.json"))
                != receipt["manifest_sha256"]):
            raise ValueError("Relocation evidence changed")
        with args.output.open("rb") as handle:
            prefix = handle.read(receipt["prefix_bytes"])
        if hashlib.sha256(prefix).hexdigest() != receipt["prefix_sha256"]:
            raise ValueError("Relocated prediction prefix changed")
        return {"status": "verified", "identity": identity,
                "copied_predictions": receipt["copied_predictions"],
                "output_dir": str(output_dir)}
    moved = prepare_relocation(source_dir / f"{identity}.jsonl", args)
    if moved is None or len(moved["predictions"]) != progress[identity]["saved"]:
        raise ValueError("Partial export count changed during relocation inspection")
    checked_protocol(moved["manifest"]["protocol"], plan["inference_protocol"])
    report = {"status": "inspected" if inspect_only else "relocated", "identity": identity,
              "copied_predictions": len(moved["predictions"]), "output_dir": str(output_dir)}
    if inspect_only:
        return report
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir) + ".lock", timeout=0):
        if source_hashes(source_dir, identity) != before:
            raise ValueError("Source test export changed before relocation")
        output_dir.mkdir(exist_ok=False)
        with (output_dir / "test_run.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        relocation = write_relocation(moved)
        if source_hashes(source_dir, identity) != before:
            raise ValueError("Source test export changed during relocation")
        receipt = {"schema": "test_storage_relocation_v1", "identity": identity,
                   "plan_sha256": plan_sha256, "source": str(source_dir),
                   "source_files": before, "destination": str(output_dir),
                   "copied_predictions": report["copied_predictions"],
                   "prefix_bytes": args.output.stat().st_size,
                   "prefix_sha256": relocation["destination_sha256"],
                   "run_sha256": sha256_file(output_dir / "test_run.json"),
                   "manifest_sha256": sha256_file(args.output.with_suffix(
                       ".jsonl.manifest.json")), "tool_sha256": sha256_file(Path(__file__))}
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("environment", "plan", "source-dir", "output-dir", "rubric"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--identity", choices=IDENTITIES, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    try:
        report = relocate_test_run(
            args.environment, args.plan, args.plan_sha256, args.identity,
            args.source_dir, args.output_dir, adapter=args.adapter, rubric=args.rubric,
            inspect_only=args.inspect)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
