"""Build the predeclared first-100 review pilot without changing saved labels."""

import argparse
import json
from pathlib import Path

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.evaluation.answer_metrics import exact_match
from vqa_explanation.preference.drafts import load_review_drafts
from vqa_explanation.preference.review import render_editor
from vqa_explanation.utils.run_provenance import sha256_file


def build_pilot(review_dir, drafts_path, recipe_path, *, count=100, refresh=False):
    review_dir = Path(review_dir)
    if not 1 <= count <= 2000:
        raise ValueError("Pilot count must be between 1 and 2,000")
    output = review_dir / f"pilot_review_{count}.html"
    manifest_path = review_dir / f"pilot_review_{count}.json"
    if output.exists() or manifest_path.exists():
        if not refresh:
            raise FileExistsError("Pilot already exists; preserve it before rebuilding")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(output) != previous["page_sha256"]:
            raise ValueError("Generated pilot page was edited; refusing to overwrite it")

    notes, drafts = load_review_drafts(review_dir, drafts_path, recipe_path)
    inputs = list(read_jsonl(review_dir / "inputs.jsonl"))
    predictions = {
        row["question_id"]: row for row in read_jsonl(review_dir / "candidates.jsonl")
    }
    selected = inputs[:count]
    ids = [row["question_id"] for row in selected]
    images = [row["image_path"] for row in selected]
    if len(set(ids)) != count or len(set(images)) != count:
        raise ValueError("Pilot must retain one unique question per unique image")
    if any(row["split"] != "train" for row in selected):
        raise ValueError("Pilot must contain train questions only")
    queue = {
        "schema": "candidate_review_queue_v1",
        "label": f"Pilot cố định {count} câu đầu của pool seed 42",
        "question_ids": ids,
    }
    source = (review_dir / "index.html").read_text(encoding="utf-8")
    start = source.index('<section id="review-editor">')
    end = source.index('</script>', source.index('(() => {', start)) + len('</script>')
    page = source[:start] + render_editor(notes, drafts=drafts, queue=queue) + source[end:]
    output.write_text(page, encoding="utf-8")
    status_counts = {
        state: sum(notes[position]["review_status"] == state for position in range(count))
        for state in ("pending", "accepted", "rejected", "uncertain")
    }
    manifest = {
        "schema": "preference_review_pilot_v1",
        "status": "pending_human_review",
        "selection_method": (
            "First N records from the immutable seed-42, train-only, one-question-per-image pool; "
            "declared before review and not selected by outcome."
        ),
        "count": count,
        "question_ids": ids,
        "unique_images": len(set(images)),
        "candidate_answer_exact_match": int(
            sum(
                exact_match(predictions[row["question_id"]]["prediction_answer"], row["answer"])
                for row in selected
            )
        ),
        "review_status_counts": status_counts,
        "source_inputs_sha256": sha256_file(review_dir / "inputs.jsonl"),
        "source_candidates_sha256": sha256_file(review_dir / "candidates.jsonl"),
        "source_notes_sha256": sha256_file(review_dir / "review_notes.jsonl"),
        "drafts_sha256": sha256_file(drafts_path),
        "recipe_sha256": sha256_file(recipe_path),
        "page_sha256": sha256_file(output),
        "limitations": (
            "Answer exact match is only a routing statistic. It is not visual verification, "
            "an accepted preference pair count, or a model-quality estimate."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = build_pilot(
        args.review_dir, args.drafts, args.recipe, count=args.count, refresh=args.refresh
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
