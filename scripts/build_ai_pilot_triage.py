"""Record image-inspected AI triage for the fixed 100-question preference pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.preference.review import render_editor
from vqa_explanation.preference.schema import AI_TRIAGE_VERDICTS, ERROR_TYPES
from vqa_explanation.utils.environment import project_root
from vqa_explanation.utils.run_provenance import sha256_file

EXPLANATION_ISSUES = {"contradiction", "unsupported_detail", "answer_support", "lexical_precision"}

def validate_source_hashes(pilot: dict, source_paths: dict[str, Path]) -> dict[str, str]:
    """Match the current review files to the hashes frozen in the pilot."""
    actual = {key: sha256_file(path) for key, path in source_paths.items()}
    for key, digest in actual.items():
        if pilot.get(key) != digest:
            raise ValueError(f"Frozen pilot source hash mismatch for {key}")
    return actual


def load_visual_review(path: Path, question_ids: list[str], candidates: dict) -> dict:
    """Check the supplied visual decisions without promoting them to human labels."""
    review = json.loads(path.read_text(encoding="utf-8"))
    if (
        review.get("schema") != "ai_visual_pilot_review_v1"
        or review.get("status") != "pending_user_review"
        or review.get("annotator_type") != "ai"
        or review.get("reviewer") is not None
        or review.get("human_verified") is not False
        or review.get("training_eligible") is not False
    ):
        raise ValueError("Visual review must remain an unnamed AI proposal")
    source = (project_root() / review["source_triage"]).resolve()
    if not source.is_relative_to(project_root().resolve()):
        raise ValueError("Visual review source escapes the project")
    if sha256_file(source) != review["source_triage_sha256"]:
        raise ValueError("Visual review source checksum mismatch")
    rows = review.get("records", [])
    if [row.get("question_id") for row in rows] != question_ids:
        raise ValueError("Visual review must cover the frozen pilot IDs in order")
    if review.get("count") != len(rows):
        raise ValueError("Visual review count mismatch")
    if review.get("verdict_counts") != dict(Counter(row.get("ai_verdict") for row in rows)):
        raise ValueError("Visual review verdict counts mismatch")
    for row in rows:
        qid = row["question_id"]
        if (
            row.get("reviewer") is not None
            or row.get("human_verified") is not False
            or row.get("training_eligible") is not False
            or row.get("ai_verdict") not in AI_TRIAGE_VERDICTS
            or row.get("answer_assessment_ai") not in {"correct", "incorrect", "uncertain"}
            or not isinstance(row.get("image_evidence_ai"), str)
            or not row["image_evidence_ai"].strip()
        ):
            raise ValueError(f"{qid}: invalid AI decision or missing image evidence")
        if row["ai_verdict"] == "repair_explanation":
            chosen, span = row.get("proposed_chosen_explanation"), row.get("unsupported_span")
            rejected = candidates[qid]["prediction_explanation"]
            if (
                row["answer_assessment_ai"] != "correct"
                or not isinstance(chosen, str) or not chosen.strip()
                or not isinstance(span, str) or not span.strip() or span not in rejected
                or chosen.strip() == rejected.strip() or row.get("error_type") not in ERROR_TYPES
            ):
                raise ValueError(f"{qid}: invalid explanation repair")
            if ("explanation_issue_ai" in row
                    and row["explanation_issue_ai"] not in EXPLANATION_ISSUES):
                raise ValueError(f"{qid}: invalid explanation issue")
    return {row["question_id"]: row for row in rows}


def build_ai_triage(
    review_dir: Path,
    pilot_path: Path,
    output_dir: Path,
    page_name: str = "pilot_ai_triage_100.html",
    assistance_mode: str = "visible",
    *,
    visual_review_path: Path,
) -> dict:
    """Validate the frozen sources and write immutable AI proposals, not labels."""
    review_dir, output_dir = review_dir.resolve(), output_dir.resolve()
    root = project_root().resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("Output directory must stay inside the project")
    if Path(page_name).name != page_name or not page_name.endswith(".html"):
        raise ValueError("page_name must be a safe HTML filename")
    if assistance_mode not in {"visible", "on_request"}:
        raise ValueError("assistance_mode must be visible or on_request")
    output_source = output_dir.relative_to(root).as_posix()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    ids = pilot.get("question_ids")
    if pilot.get("schema") != "preference_review_pilot_v1" or not isinstance(ids, list):
        raise ValueError("Expected a preference_review_pilot_v1 manifest")
    if len(ids) != 100:
        raise ValueError("This protocol is frozen for the 100-question pilot")

    inputs_path = review_dir / "inputs.jsonl"
    candidates_path = review_dir / "candidates.jsonl"
    notes_path = review_dir / "review_notes.jsonl"
    source_hashes = validate_source_hashes(
        pilot,
        {
            "source_inputs_sha256": inputs_path,
            "source_candidates_sha256": candidates_path,
            "source_notes_sha256": notes_path,
        },
    )

    candidates_manifest_path = review_dir / "candidates.jsonl.manifest.json"
    candidates_manifest = json.loads(candidates_manifest_path.read_text(encoding="utf-8"))
    frozen_image_hashes = candidates_manifest.get("inputs", {}).get("image_hashes", {})
    if not isinstance(frozen_image_hashes, dict):
        raise ValueError("Candidate manifest is missing frozen image hashes")

    inputs = list(read_jsonl(inputs_path))
    candidates = {row["question_id"]: row for row in read_jsonl(candidates_path)}
    notes = list(read_jsonl(notes_path))
    input_by_id = {row["question_id"]: row for row in inputs}
    if ids != [row["question_id"] for row in inputs[:100]]:
        raise ValueError("Pilot IDs no longer match the first 100 frozen inputs")
    if any(qid not in candidates for qid in ids):
        raise ValueError("Pilot candidate is missing")
    if len(notes) != len(inputs):
        raise ValueError("Review notes no longer align with frozen inputs")

    visual_rows = load_visual_review(visual_review_path, ids, candidates)
    if visual_rows:
        verdicts = {qid: row["ai_verdict"] for qid, row in visual_rows.items()}
        for qid, verdict in verdicts.items():
            if verdict == "repair_explanation" and (
                candidates[qid]["prediction_answer"] != input_by_id[qid]["answer"]
            ):
                raise ValueError(f"{qid}: repair needs a shared, verified answer target")

    old_drafts_path = review_dir.parent / "preference_drafts_m1_3683_20261008" / "drafts.jsonl"
    old_drafts = [
        row
        for row in read_jsonl(old_drafts_path)
        if verdicts.get(row["source_question_id"]) in {"repair_explanation", "pass_grounded"}
        and row["origin_kind"] == "synthetic"
        and visual_rows.get(row["source_question_id"], {}).get(
            "retain_existing_synthetic_draft", True
        )
    ]
    old_by_id = {row["source_question_id"]: row for row in old_drafts}
    candidate_sha = source_hashes["source_candidates_sha256"]
    input_sha = source_hashes["source_inputs_sha256"]
    pilot_sha = sha256_file(pilot_path)
    counts = Counter(verdicts.values())
    created_at = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True)
    recipe = {
        "schema": "preference_ai_pilot_triage_recipe_v1",
        "created_at_utc": created_at,
        "author_type": "ai",
        "author": "OpenAI Codex",
        "human_review_performed": False,
        "review_page_assistance_mode": assistance_mode,
        "purpose": "Image-inspected first pass that reduces human pilot review effort.",
        "source_inputs_sha256": input_sha,
        "source_candidates_sha256": candidate_sha,
        "source_notes_sha256": source_hashes["source_notes_sha256"],
        "source_candidates_manifest_sha256": sha256_file(candidates_manifest_path),
        "source_pilot_sha256": pilot_sha,
        "source_existing_drafts_sha256": sha256_file(old_drafts_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "question_ids": ids,
        "verdict_counts": dict(sorted(counts.items())),
        "constraints": [
            "Every referenced image was inspected directly.",
            "No item is marked accepted and no human reviewer is assigned.",
            "Natural rejected text remains the exact M1 explanation.",
            "Only answer-correct cases receive repair proposals.",
            "Subjective, visually unresolved and inconsistent cases are excluded from proposals.",
        ],
    }
    if visual_review_path:
        recipe["source_visual_review"] = {
            "path": str(visual_review_path.resolve().relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(visual_review_path),
        }
    recipe_path = output_dir / "recipe.json"
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    recipe_sha = sha256_file(recipe_path)

    triage_rows, new_drafts = [], []
    for position, qid in enumerate(ids, start=1):
        source, candidate, verdict = input_by_id[qid], candidates[qid], verdicts[qid]
        image_path = review_dir / source["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing pilot image: {image_path}")
        image_sha = sha256_file(image_path)
        if frozen_image_hashes.get(qid) != image_sha:
            raise ValueError(f"Frozen image hash mismatch for question {qid}")
        row = {
            "schema": "preference_ai_triage_v1",
            "status": "pending_human_review",
            "human_verified": False,
            "training_eligible": False,
            "annotator_type": "ai",
            "annotator": "OpenAI Codex",
            "reviewer": None,
            "created_at_utc": created_at,
            "source_question_id": qid,
            "source_input_position": position,
            "split": source["split"],
            "question": source["question"],
            "reference_answer": source["answer"],
            "candidate_answer": candidate["prediction_answer"],
            "candidate_explanation": candidate["prediction_explanation"],
            "reference_explanations": source["explanations"],
            "image_path": source["image_path"],
            "image_sha256": image_sha,
            "ai_inspected_image": True,
            "ai_verdict": verdict,
            "recipe_sha256": recipe_sha,
        }
        if verdict == "repair_explanation":
            repair = (
                {
                    "chosen": visual_rows[qid]["proposed_chosen_explanation"],
                    "span": visual_rows[qid]["unsupported_span"],
                    "error": visual_rows[qid]["error_type"],
                    "evidence": visual_rows[qid]["image_evidence_ai"],
                }
            )
            row.update(
                {
                    "answer_assessment_ai": "correct",
                    "proposed_chosen_explanation": repair["chosen"],
                    "proposed_rejected_explanation": candidate["prediction_explanation"],
                    "unsupported_span": repair["span"],
                    "error_type": repair["error"],
                    "image_evidence_ai": repair["evidence"],
                    "rationale": (
                        "Answer is usable; replace the explanation only after human confirmation."
                    ),
                }
            )
            if qid not in old_by_id:
                new_drafts.append(
                    {
                        "schema": "preference_ai_draft_v1",
                        "draft_id": f"ai-pilot-natural-{qid}",
                        "status": "pending_ai_draft",
                        "reviewer": None,
                        "reviewed_at": None,
                        "human_verified": False,
                        "ai_inspected_image": True,
                        "source_question_id": qid,
                        "proposed_chosen_explanation": repair["chosen"],
                        "proposed_rejected_explanation": candidate["prediction_explanation"],
                        "unsupported_span": repair["span"],
                        "error_type": repair["error"],
                        "origin_kind": "model",
                        "image_evidence_ai": repair["evidence"],
                        "chosen_provenance": {
                            "kind": "synthetic",
                            "source": f"{output_source}/triage.jsonl:{qid}",
                            "recipe_sha256": recipe_sha,
                        },
                        "rejected_provenance": {
                            "kind": "model",
                            "source": (
                                "artifacts/preference_editor_m1_3683_2000_20261008/"
                                f"candidates.jsonl:{qid}"
                            ),
                        },
                        "limitation": (
                            "AI proposal only; a person must inspect the image before acceptance."
                        ),
                    }
                )
        elif verdict == "exclude_answer_mismatch":
            row.update(
                {
                    "answer_assessment_ai": "incorrect",
                    "rationale": (
                        "Candidate answer conflicts with the frozen reference/visible "
                        "target; no explanation-only pair proposed."
                    ),
                }
            )
        elif verdict == "exclude_ambiguous":
            row.update({
                "answer_assessment_ai": "uncertain",
                "rationale": (
                    visual_rows[qid]["image_evidence_ai"]
                ),
            })
        elif verdict == "exclude_out_of_scope":
            row.update({
                "answer_assessment_ai": "correct",
                "rationale": (
                    visual_rows[qid]["image_evidence_ai"]
                ),
            })
        else:
            row.update(
                {
                    "answer_assessment_ai": "correct",
                    "rationale": (
                        "Candidate answer and explanation are visually consistent; "
                        "no repair pair is needed."
                    ),
                }
            )
        if visual_rows:
            row["answer_assessment_ai"] = visual_rows[qid]["answer_assessment_ai"]
            row["image_evidence_ai"] = visual_rows[qid]["image_evidence_ai"]
            row["rationale"] = visual_rows[qid]["image_evidence_ai"]
            if "explanation_issue_ai" in visual_rows[qid]:
                row["explanation_issue_ai"] = visual_rows[qid]["explanation_issue_ai"]
                for draft in new_drafts:
                    if draft["source_question_id"] == qid:
                        draft["explanation_issue_ai"] = row["explanation_issue_ai"]
        triage_rows.append(row)

    triage_path = output_dir / "triage.jsonl"
    triage_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in triage_rows),
        encoding="utf-8",
    )
    all_drafts = old_drafts + new_drafts
    draft_ids = [row["source_question_id"] for row in all_drafts]
    if len(draft_ids) != len(set(draft_ids)):
        raise ValueError("Combined AI drafts contain duplicate question IDs")
    drafts_path = output_dir / "drafts.jsonl"
    drafts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_drafts),
        encoding="utf-8",
    )

    queue = {
        "schema": "candidate_review_queue_v1",
        "label": "Pilot 100 câu — đã có AI triage, vẫn chờ người duyệt",
        "question_ids": ids,
    }
    source_page = (review_dir / "index.html").read_text(encoding="utf-8")
    start = source_page.index('<section id="review-editor">')
    end = source_page.index("</script>", source_page.index("(() => {", start)) + len("</script>")
    banner = (
        "<section><h2>AI triage: chỉ là đề xuất</h2><p>Đã xem trực tiếp 100 ảnh. "
        f"Kết quả: {counts['pass_grounded']} phù hợp, "
        f"{counts['repair_explanation']} có thể sửa giải thích, "
        f"{counts['exclude_answer_mismatch']} không dùng do đáp án/đích câu hỏi và "
        f"{counts['exclude_ambiguous']} mơ hồ/chủ quan, "
        f"{counts['exclude_out_of_scope']} lỗi diễn đạt ngoài phạm vi cặp bám ảnh. "
        "Tên người duyệt để trống. Bạn xem lại trước khi xác nhận cặp; có thể dùng mã "
        "ẩn danh khi xác nhận. Mọi dòng vẫn pending.</p></section>"
    )
    page = (
        source_page[:start]
        + banner
        + render_editor(
            notes,
            drafts=all_drafts,
            queue=queue,
            advisories=triage_rows,
            assistance_mode=assistance_mode,
        )
        + source_page[end:]
    )
    page_path = review_dir / page_name
    if page_path.exists():
        raise FileExistsError(f"Generated review page already exists: {page_path}")
    page_path.write_text(page, encoding="utf-8")

    summary = {
        "schema": "preference_ai_pilot_triage_summary_v1",
        "status": "pending_human_review",
        "count": len(triage_rows),
        "verdict_counts": dict(sorted(counts.items())),
        "drafts_total": len(all_drafts),
        "existing_drafts": len(old_drafts),
        "new_natural_repair_drafts": len(new_drafts),
        "natural_repair_issue_counts": dict(Counter(
            row.get("explanation_issue_ai", "unclassified")
            for row in triage_rows if row["ai_verdict"] == "repair_explanation"
        )),
        "accepted": 0,
        "human_reviewers": 0,
        "training_eligible": 0,
        "review_page_assistance_mode": assistance_mode,
        "recipe_sha256": recipe_sha,
        "triage_sha256": sha256_file(triage_path),
        "drafts_sha256": sha256_file(drafts_path),
        "page": str(page_path.relative_to(review_dir.parent.parent)).replace("\\", "/"),
        "page_sha256": sha256_file(page_path),
        "limitations": (
            "AI visual triage reduces review effort but does not replace human "
            "acceptance or validate reviewer identity."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-name", default="pilot_ai_triage_100.html")
    parser.add_argument("--visual-review", type=Path, required=True,
                        help="Source-bound AI visual decisions; no unreviewed CLI fallback")
    parser.add_argument(
        "--assistance-mode",
        choices=("visible", "on_request"),
        default="visible",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_ai_triage(
                args.review_dir,
                args.pilot,
                args.output_dir,
                args.page_name,
                args.assistance_mode,
                visual_review_path=args.visual_review,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
