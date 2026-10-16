"""Snapshot review progress and optionally freeze an explicitly reviewed subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vqa_explanation.data.io import read_jsonl, write_json, write_jsonl
from vqa_explanation.preference.bundle import verify_bundle
from vqa_explanation.preference.freeze import freeze_pairs
from vqa_explanation.preference.pilot_report import render_pilot_markdown, summarize_pilot
from vqa_explanation.preference.review import export_review, review_origin_errors
from vqa_explanation.utils.run_provenance import sha256_file, utc_now


def verify_pilot_sources(review_dir: Path, pilot: dict) -> dict[str, str]:
    if not isinstance(pilot, dict) or pilot.get("schema") != "preference_review_pilot_v1":
        raise ValueError("Expected a preference_review_pilot_v1 manifest")
    ids = pilot.get("question_ids")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(qid, str) or not qid.strip() for qid in ids)
            or len(ids) != len(set(ids))
            or type(pilot.get("count")) is not int or pilot["count"] != len(ids)):
        raise ValueError("Pilot requires unique question IDs and a matching count")
    paths = {
        "source_inputs_sha256": review_dir / "inputs.jsonl",
        "source_candidates_sha256": review_dir / "candidates.jsonl",
        "source_notes_sha256": review_dir / "review_notes.jsonl",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if any(pilot.get(key) != digest for key, digest in hashes.items()):
        raise ValueError("Frozen pilot source hash mismatch")
    return hashes


def _has_pair_draft(row: dict) -> bool:
    fields = ("proposed_chosen_explanation", "proposed_rejected_explanation",
              "unsupported_span", "error_type", "image_evidence")
    return (row.get("answer_assessment") == "correct"
            and row.get("canonical_answer_verified") is True
            and all(isinstance(row.get(key), str) and row[key].strip() for key in fields)
            and not review_origin_errors(row))


def _summary_markdown(summary: dict) -> str:
    counts, pilot = summary["review_counts"], summary["pilot"]
    lines = [
        "# Bàn giao dữ liệu preference", "",
        f"Trạng thái: {summary['status']}.", "",
        f"Pilot đã xác nhận {pilot['resolved']}/{pilot['records']} câu; "
        f"còn {len(summary['pending_pilot_ids'])} câu chưa chốt.",
        f"Cặp accepted trong toàn bộ pool: {counts.get('accepted', 0)}. "
        f"Cặp nháp trong pilot: {len(summary['pair_draft_ids'])}.", "",
        "Nháp có đủ trường không có nghĩa đã kiểm đúng theo ảnh. "
        "Không tự ghi người xác nhận, quyết định hoặc nhãn accepted.", "",
    ]
    if summary["freeze_blockers"]:
        lines += ["Chưa đóng gói dữ liệu train:", ""]
        lines += [f"- {reason}" for reason in summary["freeze_blockers"]]
        lines.append("")
    if summary["frozen_bundle"]:
        lines += [
            f"Bộ đã đóng băng: `{summary['frozen_bundle']}`, "
            f"{summary['frozen_pairs']} cặp cùng chosen cho M2/M3.", "",
            "Tiếp theo kiểm expected_pairs ở hai config, M1 selection và môi trường GPU. "
            "Lệnh này không tải model, không chạy preflight GPU hoặc train.", "",
        ]
    else:
        lines += [
            "Tiếp theo: kiểm các cặp nháp, chốt cả câu loại/chưa chắc trong pilot, "
            "xuất JSONL rồi chạy lại lệnh. Sau pilot mới chốt phạm vi mở rộng và N.", "",
        ]
    return "\n".join(lines)


def build_handoff(
    review_dir: Path, notes_path: Path, pilot_path: Path, output_dir: Path, *,
    target_pairs: int = 1000, freeze: bool = False, expected_pairs: int | None = None,
    bundle: Path | None = None, expected_bundle_sha256: str | None = None,
    annotations: Path | None = None, image_root: Path | None = None,
) -> dict:
    """Publish a new handoff directory; never mutate labels or start training."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite handoff: {output_dir}")
    if type(target_pairs) is not int or target_pairs <= 0:
        raise ValueError("target_pairs must be positive; it is a planning target only")
    if freeze and (type(expected_pairs) is not int or expected_pairs <= 0
                   or any(value is None for value in (
                       bundle, expected_bundle_sha256, annotations, image_root))):
        raise ValueError("Freeze requires expected_pairs, bundle/hash, annotations and image_root")
    if not freeze and any(value is not None for value in (
        expected_pairs, bundle, expected_bundle_sha256, annotations, image_root
    )):
        raise ValueError("Freeze arguments require explicit freeze mode")

    pilot_hash, notes_hash = sha256_file(pilot_path), sha256_file(notes_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    source_hashes = verify_pilot_sources(review_dir, pilot)
    baseline = list(read_jsonl(review_dir / "review_notes.jsonl"))
    notes = list(read_jsonl(notes_path))
    report = summarize_pilot(notes, baseline, pilot["question_ids"], target_pairs=target_pairs)
    by_id = {row.get("source_question_id"): row for row in notes}
    pilot_rows = [by_id[qid] for qid in pilot["question_ids"] if qid in by_id]
    blockers = []
    if report["status"] == "invalid":
        blockers.append("Nhãn hoặc nguồn không hợp lệ; xem lỗi trong pilot_report.json.")
    if not report["gates"]["pilot_complete"]:
        blockers.append("Pilot cố định chưa được xác nhận đầy đủ.")
    accepted = report["audit"]["counts"].get("accepted", 0)
    if accepted == 0:
        blockers.append("Chưa có cặp accepted; không dùng nháp pending làm đầu vào train.")
    if freeze and accepted != expected_pairs:
        blockers.append(f"Số accepted ({accepted}) khác expected_pairs ({expected_pairs}).")

    status = ("invalid" if report["status"] == "invalid" else "needs_pilot_review"
              if not report["gates"]["pilot_complete"] else "needs_accepted_pairs"
              if not accepted else "ready_to_lock_pair_budget")
    summary = {
        "schema": "preference_review_handoff_v1", "created_at_utc": utc_now(),
        "status": status, "review_counts": report["audit"]["counts"],
        "pilot": report["pilot"], "freeze_requested": freeze, "freeze_blockers": blockers,
        "frozen_bundle": None, "frozen_pairs": 0, "gpu_training_started": False,
        "training_gate_open": False,
        "pending_pilot_ids": [r["source_question_id"] for r in pilot_rows
                              if r.get("review_status") == "pending"],
        "pair_draft_ids": [r["source_question_id"] for r in pilot_rows
                           if r.get("review_status") == "pending" and _has_pair_draft(r)],
        "inputs": {"notes_sha256": notes_hash, "pilot_sha256": pilot_hash,
                   **source_hashes},
        "limitations": "Checks supplied metadata, not reviewer identity or visual truth. "
                        "Frozen metadata is not proof of sufficient sample size or GPU readiness.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".review-handoff-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "handoff"
        stage.mkdir()
        snapshot = stage / "notes_snapshot.jsonl"
        write_jsonl(snapshot, notes)
        if freeze and not blockers:
            data, inputs, predictions = verify_bundle(bundle, expected_bundle_sha256)
            for member, anchor in (("inputs.jsonl", "source_inputs_sha256"),
                                   ("candidates.jsonl", "source_candidates_sha256")):
                if hashlib.sha256(data[member]).hexdigest() != source_hashes[anchor]:
                    raise ValueError(f"Review bundle differs from frozen pilot: {member}")
            manifest = json.loads(data["candidates.jsonl.manifest.json"])
            pairs = export_review(snapshot, inputs, predictions, manifest,
                                  expected_bundle_sha256, annotations, image_root, stage / "export")
            frozen = freeze_pairs(pairs, annotations, image_root, stage / "frozen")
            if frozen["pairs"] != expected_pairs:
                raise ValueError("Frozen pair count differs from the declared budget")
            summary.update(status="frozen_metadata", frozen_bundle="frozen",
                           frozen_pairs=frozen["pairs"])
            summary["inputs"].update(bundle_sha256=expected_bundle_sha256,
                                     annotations_sha256=frozen["annotations_sha256"])
        report["inputs"] = summary["inputs"]
        write_json(stage / "pilot_report.json", report)
        (stage / "pilot_report.md").write_text(render_pilot_markdown(report), encoding="utf-8")
        (stage / "handoff.md").write_text(_summary_markdown(summary), encoding="utf-8")
        if (notes_hash != sha256_file(notes_path) or pilot_hash != sha256_file(pilot_path)
                or source_hashes != verify_pilot_sources(review_dir, pilot)):
            raise ValueError("Review inputs changed while creating the handoff")
        summary["files"] = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        write_json(stage / "handoff.json", summary)
        if output_dir.exists():
            raise FileExistsError(f"Handoff target appeared while publishing: {output_dir}")
        stage.rename(output_dir)
    return summary
