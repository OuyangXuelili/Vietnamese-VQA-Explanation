"""Check pair difficulty without modifying the frozen dataset."""

import argparse
import hashlib
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.inference.export import validate_saved_predictions
from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.preference.freeze import prepare_exports
from vqa_explanation.preference.quality import (
    assess_reviews,
    candidate_origin,
    pair_digest,
    review_template,
    text_only_screen,
)
from vqa_explanation.utils.run_provenance import sha256_file


def sample_sheets(rows, image_root, output_dir, font_path):
    """Show the declared sample; a contact sheet does not create review labels."""
    font = ImageFont.truetype(str(font_path), 17)
    root = image_root.resolve()
    for start in range(0, len(rows), 8):
        page = Image.new("RGB", (1800, 1200), "white")
        draw = ImageDraw.Draw(page)
        for index, row in enumerate(rows[start:start + 8]):
            path = (root / row["image_path"]).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != row["image_sha256"]:
                raise ValueError("Sample image path or hash differs from the frozen pair")
            x, y = index % 4 * 450, index // 4 * 600
            with Image.open(path) as original:
                picture = original.convert("RGB")
                picture.thumbnail((430, 285))
                page.paste(picture, (x + (450 - picture.width) // 2, y + 22))
            lines = [f'{start + index + 1}. {row["pair_id"]}']
            for label, value in (("Q", row["question"]),
                                 ("A", parse_prediction(row["chosen"]).answer),
                                 ("C", parse_prediction(row["chosen"]).explanation),
                                 ("R", parse_prediction(row["rejected"]).explanation)):
                lines.extend(textwrap.wrap(f"{label}: {value}", 48))
            draw.multiline_text((x + 8, y + 310), "\n".join(lines),
                                font=font, fill="black", spacing=3)
        page.save(output_dir / f"sample_{start:03d}.jpg", quality=92)


def sample_findings(path, sample, sample_hash):
    record = json.loads(path.read_text("utf-8"))
    columns = ["sample_index", "relevance", "text_conflict", "visual", "reason"]
    if (record.get("schema") != "preference_sample_audit_v1"
            or record.get("sample_sha256") != sample_hash or record.get("columns") != columns
            or record.get("reviewer_type") not in {"human", "ai"}):
        raise ValueError("Sample review does not match the declared sample")
    rows = record.get("rows")
    if (not isinstance(rows, list) or len(rows) != len(sample)
            or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
            or {row[0] for row in rows} != set(range(1, len(sample) + 1))):
        raise ValueError("Supply one explicit finding for each sampled pair")
    findings = []
    for values in rows:
        item = dict(zip(columns, values, strict=True))
        if (type(item["sample_index"]) is not int
                or item["relevance"] not in {"direct", "target_detail", "unrelated"}
                or item["text_conflict"] not in {"yes", "no", "uncertain"}
                or item["visual"] not in {"supported", "uncertain", "chosen_unsupported",
                                           "answer_uncertain", "both_possible"}
                or not isinstance(item["reason"], str) or not item["reason"].strip()):
            raise ValueError("Invalid per-pair audit finding")
        pair = sample[item["sample_index"] - 1]
        findings.append({**item, "pair_id": pair["pair_id"], "pair_sha256": pair_digest(pair),
                         "reviewer_type": record["reviewer_type"],
                         "independent_human_verified": record.get("independent_human_verified"),
                         "blinded": record.get("blinded")})
    summary = {key: dict(Counter(row[key] for row in findings))
               for key in ("relevance", "text_conflict", "visual")}
    summary.update(count=len(findings), review_sha256=sha256_file(path),
                   needs_followup=sum(row["relevance"] == "unrelated"
                                      or row["text_conflict"] != "no"
                                      or row["visual"] != "supported" for row in findings),
                   limitations=record.get("limitations"), training_labels_created=0)
    return findings, summary


def recover_lineage(pairs, annotations, candidates_path, selection_path, batches):
    canonical = {row["question_id"]: row for row in read_jsonl(annotations)}
    candidates = list(read_jsonl(candidates_path))
    manifest_path = Path(str(candidates_path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    origin = candidate_origin(manifest["inputs"])
    selection = json.loads(selection_path.read_text("utf-8"))
    if (selection.get("status") != "selected" or selection.get("selection_split") != "validation"
            or selection.get("model_id") != origin["model_id"]
            or selection.get("model_revision") != origin["model_revision"]
            or selection.get("adapter", {}).get("sha256") != origin["checkpoint_sha256"]
            or selection.get("adapter", {}).get("files") != manifest["inputs"]["adapter_files"]
            or selection.get("annotations_sha256") != sha256_file(annotations)):
        raise ValueError("Candidate source differs from the saved selected M1 identity")
    validate_saved_predictions(candidates, list(canonical.values()), manifest)
    by_id = {row["question_id"]: row for row in candidates}
    candidate_hash, manifest_hash = sha256_file(candidates_path), sha256_file(manifest_path)
    reviewed_items = {}
    for batch in sorted(batches.glob("batch_*")):
        sheet = json.loads((batch / "manifest.json").read_text("utf-8"))
        if (sheet.get("candidates_sha256") != candidate_hash
                or sheet.get("items_sha256") != sha256_file(batch / "items.jsonl")
                or sheet.get("sheet_sha256") != sha256_file(batch / "images.png")):
            raise ValueError("Historical review sheet hash differs from saved inputs")
        for row in read_jsonl(batch / "items.jsonl"):
            if row["question_id"] in reviewed_items:
                raise ValueError("Duplicate question in historical review sheets")
            reviewed_items[row["question_id"]] = row
    result = []
    for pair in pairs:
        candidate = by_id.get(pair["source_question_id"])
        item = reviewed_items.get(pair["source_question_id"])
        if (candidate is None or item is None or candidate["split"] != "train"
                or any(candidate[key] != pair[key] for key in
                       ("image_id", "image_path", "image_sha256", "question"))
                or item.get("candidate_raw") != candidate["raw_output"]
                or item.get("image_sha256") != candidate["image_sha256"]):
            raise ValueError("A pair cannot be linked to its original reviewed M1 candidate")
        result.append({"pair_id": pair["pair_id"], "pair_sha256": pair_digest(pair),
                       "original_candidate_origin": {
                           **origin, "candidates_sha256": candidate_hash,
                           "candidate_manifest_sha256": manifest_hash,
                           "candidate_raw_sha256": hashlib.sha256(
                               candidate["raw_output"].encode()).hexdigest(),
                       }})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--quality-review", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=500)
    parser.add_argument("--required-review-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--sample-review", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--m1-selection", type=Path)
    parser.add_argument("--review-batches", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; preserve the previous audit")
    if bool(args.annotations) != bool(args.image_root):
        parser.error("Supply annotations and image-root together")
    if args.font and not args.image_root:
        parser.error("Contact sheets require image-root and annotations")
    lineage_args = (args.candidates, args.m1_selection, args.review_batches)
    if any(lineage_args) and (not all(lineage_args) or not args.annotations):
        parser.error("Lineage recovery requires candidates, m1-selection, "
                     "review-batches, annotations")
    pairs_hash = sha256_file(args.pairs)
    annotations_hash = sha256_file(args.annotations) if args.annotations else None
    pairs = list(read_jsonl(args.pairs))
    if not 1 <= args.sample_count <= len(pairs):
        parser.error("Sample count must be between one and the number of pairs")
    integrity = None
    if args.annotations:
        accepted, chosen, _ = prepare_exports(
            pairs, list(read_jsonl(args.annotations)), args.image_root,
            annotations_hash,
        )
        chosen_path = args.pairs.parent / "chosen_only.jsonl"
        integrity = {"canonical_split_image_hash_checks": "passed",
                     "accepted_export_matches": accepted == pairs,
                     "chosen_export_matches": chosen == list(read_jsonl(chosen_path))}
        if not all(integrity[key] for key in ("accepted_export_matches", "chosen_export_matches")):
            raise ValueError("Frozen M2/M3 exports differ from canonical inputs")
    reviews = list(read_jsonl(args.quality_review)) if args.quality_review else []
    review_hash = sha256_file(args.quality_review) if args.quality_review else None
    readiness = assess_reviews(pairs, reviews, args.minimum_pairs,
                               required_version=args.required_review_version)
    sample = random.Random(args.sample_seed).sample(
        sorted(pairs, key=lambda row: row["pair_id"]), args.sample_count
    )
    sample_hash = hashlib.sha256("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in sample).encode()).hexdigest()
    findings, findings_summary = None, None
    if args.sample_review:
        findings, findings_summary = sample_findings(args.sample_review, sample, sample_hash)
    lineage = None
    if args.candidates:
        lineage = recover_lineage(pairs, args.annotations, args.candidates,
                                  args.m1_selection, args.review_batches)
    screen = text_only_screen(pairs)
    if sha256_file(args.pairs) != pairs_hash or (
        args.quality_review and sha256_file(args.quality_review) != review_hash
    ) or (args.annotations and sha256_file(args.annotations) != annotations_hash):
        raise ValueError("Quality inputs changed while checking")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "text_only.json", screen)
    write_json(args.output_dir / "readiness.json", readiness)
    if not args.quality_review:
        write_jsonl(args.output_dir / "quality_review.jsonl", review_template(pairs))
    pending = set(readiness["needs_review"]) | set(readiness["missing"])
    templates = {r["pair_id"]: r for r in review_template(pairs)}
    write_jsonl(args.output_dir / "review_queue.jsonl", [
        {"pair": row, "review": templates[row["pair_id"]]}
        for row in pairs if row["pair_id"] in pending
    ])
    write_jsonl(args.output_dir / "visual_sample.jsonl", sample)
    if args.font:
        sample_sheets(sample, args.image_root, args.output_dir, args.font)
    if findings is not None:
        write_jsonl(args.output_dir / "sample_findings.jsonl", findings)
    if lineage is not None:
        write_jsonl(args.output_dir / "candidate_lineage.jsonl", lineage)
    result = {"pairs_sha256": pairs_hash, "quality_review_sha256": review_hash,
              "pair_accuracy": screen["pair_accuracy"],
              "image_bootstrap_ci95": screen["image_bootstrap_ci95"],
              "review_records_passed": readiness["ready"],
              "verification_scope": readiness["verification_scope"],
              "integrity": integrity,
              "candidate_lineage_missing": sum(not isinstance(
                  row.get("provenance", {}).get("original_candidate_origin"), dict)
                  for row in pairs),
              "chosen_sources": dict(Counter(row["provenance"]["chosen"]["kind"]
                                              for row in pairs)),
              "rejected_sources": dict(Counter(row["provenance"]["rejected"]["kind"]
                                                for row in pairs)),
              "sample": {"count": len(sample), "seed": args.sample_seed,
                         "selection": "simple_random_without_replacement",
                         "sha256": sample_hash},
              "sample_findings": findings_summary,
              "candidate_lineage_recovered": len(lineage) if lineage is not None else None,
              "output_dir": str(args.output_dir)}
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
