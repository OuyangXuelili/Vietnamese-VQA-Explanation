"""Build a draft-review page next to existing images without changing saved labels."""

import argparse
import json

from vqa_explanation.preference.drafts import load_review_drafts
from vqa_explanation.preference.review import render_editor
from vqa_explanation.utils.run_provenance import sha256_file


def attach_drafts(review_dir, drafts, recipe, *, refresh=False):
    output = review_dir / "draft_review.html"
    report = review_dir / "draft_review_manifest.json"
    if output.exists() or report.exists():
        if not refresh:
            raise FileExistsError("Draft review already exists; preserve it before rebuilding")
        previous = json.loads(report.read_text(encoding="utf-8"))
        if sha256_file(output) != previous["page_sha256"]:
            raise ValueError("Generated page was edited; refusing to overwrite it")
    notes, proposals = load_review_drafts(review_dir, drafts, recipe)
    source = (review_dir / "index.html").read_text(encoding="utf-8")
    start = source.index('<section id="review-editor">')
    end = source.index('</script>', source.index('(() => {', start)) + len('</script>')
    page = source[:start] + render_editor(notes, drafts=proposals) + source[end:]
    with output.open("w" if refresh else "x", encoding="utf-8") as stream:
        stream.write(page)
    result = {
        "status": "pending_human_review", "candidates": len(notes), "ai_drafts": len(proposals),
        "drafts_sha256": sha256_file(drafts), "recipe_sha256": sha256_file(recipe),
        "notes_sha256": sha256_file(review_dir / "review_notes.jsonl"),
        "page_sha256": sha256_file(output), "images_copied": 0,
        "accepted": sum(row["review_status"] == "accepted" for row in notes),
    }
    with report.open("w" if refresh else "x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh an unmodified generated page; saved labels stay unchanged")
    args = parser.parse_args()
    print(json.dumps(attach_drafts(args.review_dir, args.drafts, args.recipe,
                                  refresh=args.refresh), indent=2))


if __name__ == "__main__":
    main()
