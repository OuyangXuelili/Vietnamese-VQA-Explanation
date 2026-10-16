"""Copy a verified export to a new storage path without changing predictions."""

import hashlib
import json
from pathlib import Path

from vqa_explanation.inference.export import (
    export_inputs,
    read_jsonl,
    select_records,
    validate_saved_predictions,
    validate_selected_records,
)
from vqa_explanation.utils.run_provenance import sha256_file


def prepare_relocation(source, args):
    source, destination = Path(source).resolve(), args.output.resolve()
    if (source.parent.is_relative_to(destination.parent)
            or destination.parent.is_relative_to(source.parent) or destination.parent.exists()):
        raise ValueError("Relocation requires a new, separate output directory")
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        if source.exists():
            raise ValueError("Cannot relocate predictions without their manifest")
        return None
    original = json.loads(manifest_path.read_text("utf-8"))
    if (original.get("schema_version") != 1 or not isinstance(original.get("inputs"), dict)
            or not isinstance(original.get("protocol"), dict) or not original["protocol"]):
        raise ValueError("Unsupported original export manifest")
    rows = select_records(read_jsonl(args.annotations), args.split, args.limit, args.seed)
    validate_selected_records(rows, args.image_root)
    hashes = {row["question_id"]: sha256_file(args.image_root / row["image_path"])
              for row in rows}
    current = export_inputs(args, rows, hashes)
    old = original["inputs"]
    changed = {key for key in old.keys() | current.keys() if old.get(key) != current.get(key)}
    if changed - {"adapter"}:
        raise ValueError(f"Relocation changes more than storage: {sorted(changed)}")
    if "adapter" in changed and (not old.get("adapter") or not current.get("adapter")
                                 or not current.get("adapter_files")):
        raise ValueError("Adapter relocation requires matching weight evidence")
    predictions = read_jsonl(source) if source.is_file() else []
    if source.is_file() and source.stat().st_size:
        with source.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                raise ValueError("Original export has an incomplete final line")
    validate_saved_predictions(predictions, rows, original)
    relocated = [{**row, "adapter": current["adapter"]} for row in predictions]
    manifest = {**original, "inputs": current}
    validate_saved_predictions(relocated, rows, manifest)
    return {
        "source": source, "source_manifest": manifest_path, "destination": destination,
        "manifest": manifest, "predictions": relocated,
        "receipt": {
            "schema": "export_storage_relocation_v1",
            "source": str(source), "source_file_present": source.is_file(),
            "source_sha256": sha256_file(source) if source.is_file() else None,
            "source_manifest_sha256": sha256_file(manifest_path),
            "destination": str(destination), "copied_predictions": len(predictions),
            "changed_input_fields": sorted(changed),
            "original_adapter": old["adapter"], "current_adapter": current["adapter"],
            "prediction_content_changed": False,
        },
    }


def write_relocation(plan):
    receipt = plan["receipt"]
    source = plan["source"]
    if ((sha256_file(source) if source.is_file() else None) != receipt["source_sha256"]
            or sha256_file(plan["source_manifest"]) != receipt["source_manifest_sha256"]):
        raise ValueError("Original export changed after inspection")
    if (receipt.get("snapshot_receipt_sha256") is not None
            and sha256_file(source.parent / "snapshot.json")
            != receipt["snapshot_receipt_sha256"]):
        raise ValueError("Snapshot receipt changed after inspection")
    destination = plan["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in plan["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with destination.with_suffix(destination.suffix + ".manifest.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(plan["manifest"], handle, ensure_ascii=False, indent=2)
    receipt = {**receipt, "destination_sha256": sha256_file(destination)}
    with destination.with_suffix(destination.suffix + ".relocation.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    return receipt


def prepare_snapshot_recovery(source, args):
    """Prefer the longest verified, closed prefix without altering the original."""
    source = Path(source).resolve()
    plans = []
    original = prepare_relocation(source, args)
    if original is not None:
        plans.append(original)
    for receipt_path in sorted((source.parent / "recovery").glob(
        f"{source.stem}_*/snapshot.json"
    )):
        receipt = json.loads(receipt_path.read_text("utf-8"))
        snapshot = receipt_path.parent / source.name
        manifest = snapshot.with_suffix(snapshot.suffix + ".manifest.json")
        if (receipt.get("schema") != "closed_export_snapshot_v1"
                or receipt.get("output") != str(source)
                or receipt.get("snapshot") != str(snapshot)
                or receipt.get("sha256") != sha256_file(snapshot)
                or receipt.get("manifest_sha256") != sha256_file(manifest)):
            raise ValueError("Recovery snapshot receipt or file changed")
        plan = prepare_relocation(snapshot, args)
        if plan is None or len(plan["predictions"]) != receipt.get("predictions"):
            raise ValueError("Recovery snapshot count changed")
        plan["receipt"]["snapshot_receipt_sha256"] = sha256_file(receipt_path)
        plans.append(plan)
    if not plans:
        return None
    longest = max(plans, key=lambda plan: len(plan["predictions"]))
    for plan in plans:
        rows = plan["predictions"]
        if rows != longest["predictions"][:len(rows)]:
            raise ValueError("Recovery exports contain conflicting prediction prefixes")
        if plan["manifest"] != longest["manifest"]:
            raise ValueError("Recovery inference manifests differ")
    return longest


def verify_relocation_group(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    record = json.loads((output_dir / "relocation_group.json").read_text("utf-8"))
    if (record.get("schema") != "validation_export_relocation_v1"
            or record.get("source_dir") != str(source_dir) or not record.get("receipts")):
        raise ValueError("Existing destination has no matching relocation record")
    for receipt in record["receipts"]:
        source = Path(receipt["source"])
        destination = Path(receipt["destination"])
        if (source.parent != source_dir or destination.parent != output_dir
                or source.name not in {"m0.jsonl", "m1.jsonl", "m2.jsonl", "m3.jsonl"}
                or source.name != destination.name):
            raise ValueError("Relocation record has unexpected paths")
        manifest = source.with_suffix(source.suffix + ".manifest.json")
        if ((sha256_file(source) if source.is_file() else None) != receipt["source_sha256"]
                or sha256_file(manifest) != receipt["source_manifest_sha256"]):
            raise ValueError("Original export changed since relocation")
    return record


def snapshot_export(output, records):
    """Close a verified copy while the main Drive file is still being written."""
    output = Path(output)
    payload = output.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        return None
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    validate_saved_predictions(rows, records, manifest)
    digest = hashlib.sha256(payload).hexdigest()
    folder = output.parent / "recovery" / f"{output.stem}_{len(rows):04d}"
    if folder.exists():
        if sha256_file(folder / output.name) != digest:
            raise ValueError("Recovery snapshot already contains different predictions")
        return None
    folder.mkdir(parents=True)
    with (folder / output.name).open("xb") as handle:
        handle.write(payload)
    with (folder / manifest_path.name).open("xb") as handle:
        handle.write(manifest_bytes)
    receipt = {"schema": "closed_export_snapshot_v1", "predictions": len(rows),
               "output": str(output), "snapshot": str(folder / output.name),
               "sha256": digest, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    with (folder / "snapshot.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    return receipt
