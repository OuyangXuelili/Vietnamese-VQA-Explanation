"""Reuse complete, pinned M0/M1 exports without loading either model."""

import hashlib
import json
import re
from pathlib import Path

from vqa_explanation.inference.export import (
    export_inputs,
    read_jsonl,
    select_records,
    validate_saved_predictions,
    validate_selected_records,
)
from vqa_explanation.utils.run_provenance import sha256_file

BASELINE_FILES = {
    f"{arm}.jsonl{suffix}" for arm in ("m0", "m1") for suffix in ("", ".manifest.json")
}


def prepare_baseline_reuse(source_dir, expected_hashes, arguments):
    """Verify both baselines and any existing destination before writing anything."""
    if (not isinstance(expected_hashes, dict) or set(expected_hashes) != BASELINE_FILES
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in expected_hashes.values())):
        raise ValueError("Baseline reuse requires SHA-256 for M0/M1 predictions and manifests")
    if set(arguments) != {"m0", "m1"}:
        raise ValueError("Only M0/M1 can be reused as fixed baselines")
    source_dir = Path(source_dir).resolve()
    output_dir = arguments["m0"].output.parent.resolve()
    if (source_dir.is_relative_to(output_dir) or output_dir.is_relative_to(source_dir)
            or arguments["m1"].output.parent.resolve() != output_dir):
        raise ValueError("Baseline source and destination must be separate directories")
    payloads = {}
    for name, expected in expected_hashes.items():
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked baseline file: {name}")
        payloads[name] = path.read_bytes()
        if hashlib.sha256(payloads[name]).hexdigest() != expected:
            raise ValueError(f"Baseline checksum mismatch: {name}")
    files, arms, protocols = {}, {}, []
    for arm, args in arguments.items():
        if (args.experiment_id != {"m0": "M0_PRETRAINED", "m1": "M1_SFT"}[arm]
                or args.output.name != f"{arm}.jsonl" or args.split != "validation"
                or args.limit != 0):
            raise ValueError("Baseline reuse requires complete M0/M1 validation exports")
        name = f"{arm}.jsonl"
        manifest_name = name + ".manifest.json"
        original = json.loads(payloads[manifest_name])
        if (not isinstance(original, dict) or original.get("schema_version") != 1
                or not isinstance(original.get("inputs"), dict)
                or not isinstance(original.get("protocol"), dict) or not original["protocol"]):
            raise ValueError(f"Unsupported baseline manifest: {arm}")
        protocol = original["protocol"]
        generation = protocol.get("generation_config", {})
        if (any(protocol.get(key) != getattr(args, key) for key in (
            "min_pixels", "max_pixels", "max_new_tokens", "load_in_4bit",
        )) or not isinstance(protocol.get("prompt_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", protocol["prompt_sha256"])
                or not isinstance(generation, dict) or generation.get("do_sample") is not False
                or generation.get("max_new_tokens") != args.max_new_tokens):
            raise ValueError(f"Unexpected baseline inference protocol: {arm}")
        protocols.append(protocol)
        records = select_records(read_jsonl(args.annotations), args.split, args.limit, args.seed)
        validate_selected_records(records, args.image_root)
        hashes = {row["question_id"]: sha256_file(args.image_root / row["image_path"])
                  for row in records}
        current = export_inputs(args, records, hashes)
        old = original["inputs"]
        changed = {key for key in old.keys() | current.keys() if old.get(key) != current.get(key)}
        if changed - {"adapter"} or ("adapter" in changed and (
            not old.get("adapter") or not current.get("adapter")
            or not current.get("adapter_files")
        )):
            raise ValueError(f"{arm}: baseline inference identity changed: {sorted(changed)}")
        if not payloads[name].endswith(b"\n"):
            raise ValueError(f"Incomplete baseline final line: {arm}")
        predictions = [json.loads(line) for line in payloads[name].decode("utf-8").splitlines()
                       if line.strip()]
        completed = validate_saved_predictions(predictions, records, original)
        if not records or len(completed) != len(records):
            raise ValueError(f"Baseline export is incomplete: {arm}")
        relocated = [{**row, "adapter": current["adapter"]} for row in predictions]
        manifest = {**original, "inputs": current}
        validate_saved_predictions(relocated, records, manifest)
        files[name] = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in relocated)
                       .encode("utf-8") if changed else payloads[name])
        files[manifest_name] = (json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
                                if changed else payloads[manifest_name])
        arms[arm] = {"predictions": len(completed), "changed_input_fields": sorted(changed),
                     "prediction_content_changed": False}
    if protocols[0] != protocols[1]:
        raise ValueError("M0/M1 baseline inference protocols differ")
    receipt = {
        "schema": "validation_baseline_reuse_v1", "source_dir": str(source_dir),
        "source_sha256": expected_hashes,
        "destination_sha256": {name: hashlib.sha256(value).hexdigest()
                               for name, value in files.items()},
        "arms": arms,
    }
    plan = {"source_dir": source_dir, "output_dir": output_dir, "files": files,
            "receipt": receipt}
    verify_destination(plan)
    return plan


def verify_destination(plan):
    output_dir = plan["output_dir"]
    for name, expected in plan["receipt"]["destination_sha256"].items():
        path = output_dir / name
        if path.is_symlink() or (path.exists() and (
            not path.is_file() or sha256_file(path) != expected
        )):
            raise ValueError(f"Existing baseline destination differs: {name}")
    receipt_path = output_dir / "baseline_reuse.json"
    if receipt_path.is_symlink() or (receipt_path.exists() and (
        not receipt_path.is_file() or json.loads(receipt_path.read_text("utf-8")) != plan["receipt"]
    )):
        raise ValueError("Existing baseline reuse receipt differs")


def write_baseline_reuse(plan):
    """Copy only verified baselines; preserve existing matching files on resume."""
    for name, digest in plan["receipt"]["source_sha256"].items():
        path = plan["source_dir"] / name
        if path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"Baseline source changed after inspection: {name}")
    verify_destination(plan)
    plan["output_dir"].mkdir(parents=True, exist_ok=True)
    files = {**plan["files"], "baseline_reuse.json": json.dumps(
        plan["receipt"], ensure_ascii=False, indent=2,
    ).encode("utf-8")}
    for name, payload in files.items():
        path = plan["output_dir"] / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
    return plan["receipt"]
