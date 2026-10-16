"""Validate saved predictions before continuing an interrupted export."""

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any

from vqa_explanation.evaluation.answer_metrics import normalize_answer
from vqa_explanation.evaluation.comparison import EXPERIMENT_IDS
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file


def export_inputs(args, records: list[dict], image_hashes: dict[str, str]) -> dict:
    adapter_files = None
    if args.adapter:
        adapter = args.adapter.resolve()
        weights = [adapter / "adapter_model.safetensors", adapter / "adapter_model.bin"]
        files = [adapter / "adapter_config.json", *[path for path in weights if path.is_file()]]
        if len(files) < 2 or any(not path.is_file() or not path.stat().st_size for path in files):
            raise ValueError("Adapter requires a nonempty config and weight file")
        adapter_files = {path.name: sha256_file(path) for path in files}
    root = project_root()
    source_files = [
        "scripts/run_m0.py", "pyproject.toml",
        "src/vqa_explanation/inference/export.py",
        "src/vqa_explanation/inference/prompting.py",
        "src/vqa_explanation/inference/qwen25_vl.py",
        "src/vqa_explanation/training/sft.py",
    ]
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    inputs = {
        "model_id": args.model_id, "model_revision": args.revision,
        "experiment_id": args.experiment_id,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "adapter_files": adapter_files, "split": args.split, "limit": args.limit,
        "seed": args.seed, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, "load_in_4bit": args.load_in_4bit,
        "annotations_sha256": sha256_file(args.annotations),
        "selected_sha256": hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "image_hashes": image_hashes,
        "source_files": {name: sha256_file(root / name) for name in source_files},
        "python": platform.python_version(), "versions": versions,
    }
    context = getattr(args, "export_context", None)
    if context is not None:
        inputs["context"] = context
    return inputs


def validate_saved_predictions(
    saved: list[dict], records: list[dict], manifest: dict
) -> set[str]:
    expected = {row["question_id"]: row for row in records}
    completed = set()
    inputs, protocol = manifest["inputs"], manifest["protocol"]
    for row in saved:
        qid = row.get("question_id")
        if not isinstance(qid, str) or qid not in expected or qid in completed:
            raise ValueError(f"Saved prediction has an unknown or duplicate ID: {qid!r}")
        reference = expected[qid]
        fields = {
            **protocol,
            "experiment_id": inputs["experiment_id"], "adapter": inputs["adapter"],
            "model_id": inputs["model_id"], "model_revision": inputs["model_revision"],
            "seed": inputs["seed"], "question_id": qid,
            "image_sha256": inputs["image_hashes"][qid],
            "reference_answer": reference["answer"],
            "reference_explanations": reference["explanations"],
            **{key: reference[key] for key in ("image_id", "image_path", "question", "split")},
        }
        for field, value in fields.items():
            if row.get(field) != value:
                raise ValueError(f"Saved prediction {qid}: {field} changed")
        if not isinstance(row.get("raw_output"), str):
            raise ValueError(f"Saved prediction {qid}: raw_output must be a string")
        parsed = parse_prediction(row["raw_output"])
        if (row.get("prediction_answer") != parsed.answer
                or row.get("prediction_explanation") != parsed.explanation
                or type(row.get("format_valid")) is not bool
                or row["format_valid"] != parsed.format_valid):
            raise ValueError(f"Saved prediction {qid}: parsed output changed")
        latency = row.get("latency_seconds")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"Saved prediction {qid}: invalid latency")
        completed.add(qid)
    return completed



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} must contain a JSON object")
            records.append(record)
    return records


def select_records(
    records: list[dict[str, Any]], split: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    selected = [record for record in records if record.get("split") == split]
    random.Random(seed).shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return selected


def validate_selected_records(records: list[dict[str, Any]], image_root: Path) -> None:
    """Reject malformed inputs before loading the model or writing a partial export."""
    ids = []
    root = image_root.resolve()
    for index, record in enumerate(records, start=1):
        for field in ("question_id", "image_id", "image_path", "question", "answer"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Selected record {index}: {field} must be a nonempty string")
        if not normalize_answer(record["answer"]):
            raise ValueError(f"Selected record {index}: answer is empty after normalization")
        explanations = record.get("explanations")
        if not isinstance(explanations, list) or not explanations or any(
            not isinstance(value, str) or not value.strip() for value in explanations
        ):
            raise ValueError(f"Selected record {index}: explanations must be nonempty strings")
        image_path = (root / record["image_path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Selected record {index}: image path escapes image root")
        if not image_path.is_file():
            raise ValueError(f"Selected record {index}: missing image: {image_path}")
        ids.append(record["question_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Selected question IDs must be unique")


def export_predictions(args, *, runner_factory=None, input_builder=None) -> None:
    runner_factory = runner_factory or Qwen25VLRunner
    input_builder = input_builder or export_inputs
    resume = getattr(args, "resume", False)
    if not 0 < args.min_pixels <= args.max_pixels or args.max_new_tokens <= 0:
        raise SystemExit("Require 0 < min_pixels <= max_pixels and max_new_tokens > 0")
    if args.output.exists() and not resume:
        raise SystemExit(f"Output exists; choose a new path: {args.output}")
    if args.experiment_id not in EXPERIMENT_IDS:
        raise SystemExit(f"Unknown experiment: {args.experiment_id!r}")
    if args.adapter and args.experiment_id == "M0_PRETRAINED":
        raise SystemExit("An adapter run must have a distinct --experiment-id")
    if not args.adapter and args.experiment_id != "M0_PRETRAINED":
        raise SystemExit(f"{args.experiment_id} requires --adapter")
    records = select_records(read_jsonl(args.annotations), args.split, args.limit, args.seed)
    if not records:
        raise SystemExit(f"No records found for split={args.split!r}")
    try:
        validate_selected_records(records, args.image_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    image_hashes = {
        row["question_id"]: sha256_file(args.image_root / row["image_path"]) for row in records
    }
    try:
        inputs = input_builder(args, records, image_hashes)
        saved = read_jsonl(args.output) if args.output.exists() else []
        manifest = None
        completed = set()
        if resume:
            if not manifest_path.is_file():
                raise ValueError("Resume requires the export's original manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                raise ValueError("Unsupported export manifest")
            if manifest.get("inputs") != inputs:
                raise ValueError("Resume inputs/source/adapter/environment changed")
            if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"]:
                raise ValueError("Missing inference protocol in manifest")
            completed = validate_saved_predictions(saved, records, manifest)
            if saved:
                with args.output.open("rb") as source:
                    source.seek(-1, 2)
                    if source.read(1) != b"\n":
                        raise ValueError(
                            "Export has an incomplete final line; preserve and inspect it"
                        )
            print(f"Verified export: {len(completed)}/{len(records)} predictions already saved")
            if len(completed) == len(records):
                return
        elif manifest_path.exists():
            raise ValueError("Manifest exists; use --resume or choose a new output path")
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error

    runner = runner_factory(
        args.model_id,
        args.revision,
        load_in_4bit=args.load_in_4bit,
        adapter_path=str(args.adapter) if args.adapter else None,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    protocol = runner.protocol_metadata(args.max_new_tokens)
    if manifest and manifest["protocol"] != protocol:
        raise SystemExit("Loaded inference protocol changed; refusing to mix predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not manifest:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "inputs": inputs, "protocol": protocol}, handle,
                      ensure_ascii=False, indent=2)
    with args.output.open("a" if resume else "x", encoding="utf-8") as destination:
        for index, record in enumerate(records, start=1):
            if record["question_id"] in completed:
                continue
            started = time.perf_counter()
            raw = runner.generate(
                args.image_root / record["image_path"],
                record["question"],
                max_new_tokens=args.max_new_tokens,
            )
            parsed = parse_prediction(raw)
            result = {
                **protocol,
                "experiment_id": args.experiment_id,
                "adapter": inputs["adapter"],
                "max_pixels": args.max_pixels,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "seed": args.seed,
                "question_id": record["question_id"],
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "image_sha256": image_hashes[record["question_id"]],
                "split": record["split"],
                "question": record["question"],
                "reference_answer": record["answer"],
                "reference_explanations": record["explanations"],
                "prediction_answer": parsed.answer,
                "prediction_explanation": parsed.explanation,
                "format_valid": parsed.format_valid,
                "raw_output": raw,
                "latency_seconds": round(time.perf_counter() - started, 4),
            }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"[{index}/{len(records)}] {record['question_id']} format={parsed.format_valid}")
