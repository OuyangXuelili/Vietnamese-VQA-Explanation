"""Create a reproducible progress and yield report for a fixed review pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl, write_json
from vqa_explanation.preference.handoff import verify_pilot_sources
from vqa_explanation.preference.pilot_report import render_pilot_markdown, summarize_pilot
from vqa_explanation.utils.run_provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--triage", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pairs", type=int, default=1000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot report: {args.output_dir}")
    if args.target_pairs <= 0:
        raise ValueError("target-pairs must be positive")
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    if pilot.get("schema") != "preference_review_pilot_v1":
        raise ValueError("Expected a preference_review_pilot_v1 manifest")
    pilot_ids = pilot.get("question_ids")
    if not isinstance(pilot_ids, list) or not pilot_ids:
        raise ValueError("Pilot manifest is missing question IDs")
    source_hashes = verify_pilot_sources(args.review_dir, pilot)

    baseline_path = args.review_dir / "review_notes.jsonl"
    baseline = list(read_jsonl(baseline_path))
    notes = list(read_jsonl(args.notes))
    triage_rows = list(read_jsonl(args.triage)) if args.triage else None
    report = summarize_pilot(
        notes,
        baseline,
        pilot_ids,
        triage_rows=triage_rows,
        target_pairs=args.target_pairs,
    )
    report["inputs"] = {
        **source_hashes,
        "notes": str(args.notes),
        "notes_sha256": sha256_file(args.notes),
        "pilot": str(args.pilot),
        "pilot_sha256": sha256_file(args.pilot),
        "triage": str(args.triage) if args.triage else None,
        "triage_sha256": sha256_file(args.triage) if args.triage else None,
    }

    args.output_dir.mkdir(parents=True)
    json_path = args.output_dir / "pilot_report.json"
    markdown_path = args.output_dir / "pilot_report.md"
    write_json(json_path, report)
    markdown_path.write_text(render_pilot_markdown(report), encoding="utf-8")
    result = {
        "status": report["status"],
        "pilot_complete": report["gates"]["pilot_complete"],
        "training_gate_open": report["gates"]["training_gate_open"],
        "json": str(json_path),
        "json_sha256": sha256_file(json_path),
        "markdown": str(markdown_path),
        "markdown_sha256": sha256_file(markdown_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if report["status"] == "invalid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
