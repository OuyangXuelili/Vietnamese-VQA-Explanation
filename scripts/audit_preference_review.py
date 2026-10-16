"""Audit an exported review JSONL without creating training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.preference.audit import audit_review_notes
from vqa_explanation.utils.run_provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_path = args.review_dir / "review_notes.jsonl"
    baseline = list(read_jsonl(baseline_path))
    notes = list(read_jsonl(args.notes))
    pilot_ids = None
    pilot_sha256 = None
    if args.pilot is not None:
        pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
        if pilot.get("schema") != "preference_review_pilot_v1":
            raise ValueError("Expected a preference_review_pilot_v1 manifest")
        if pilot.get("source_notes_sha256") != sha256_file(baseline_path):
            raise ValueError("Pilot belongs to a different review baseline")
        pilot_ids = pilot.get("question_ids")
        if not isinstance(pilot_ids, list):
            raise ValueError("Pilot manifest is missing question IDs")
        pilot_sha256 = sha256_file(args.pilot)

    report = audit_review_notes(notes, baseline, pilot_ids)
    report["inputs"] = {
        "notes": str(args.notes),
        "notes_sha256": sha256_file(args.notes),
        "baseline_sha256": sha256_file(baseline_path),
        "pilot_sha256": pilot_sha256,
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite audit report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
