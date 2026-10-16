"""Summarize a fixed preference-review pilot without creating training labels."""

from __future__ import annotations

import math
from collections import Counter

from vqa_explanation.preference.audit import audit_review_notes
from vqa_explanation.preference.schema import AI_TRIAGE_VERDICTS, reviewer_type


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if total <= 0 or successes < 0 or successes > total:
        return None
    rate = successes / total
    scale = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / scale
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / scale
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _project_pool(target_pairs: int, rate: float, interval: list[float]) -> dict | None:
    if target_pairs <= 0 or rate <= 0:
        return None
    low, high = interval
    return {
        "target_pairs": target_pairs,
        "point_estimate_candidates": math.ceil(target_pairs / rate),
        "interval_candidates": [
            math.ceil(target_pairs / high),
            math.ceil(target_pairs / low) if low > 0 else None,
        ],
        "interpretation": (
            "Planning range derived from the completed pilot, not a quota and not a guarantee."
        ),
    }


def _rejected_origin(row: dict) -> str:
    """Mirror the exporter: unchanged M1 text retains model provenance."""
    text = row.get("proposed_rejected_explanation")
    if isinstance(text, str) and text.strip() == row.get("candidate_explanation"):
        return "model"
    return "synthetic" if row.get("rejected_origin_kind") == "synthetic" else "manual"


def summarize_pilot(
    notes: list[dict],
    baseline: list[dict],
    pilot_ids: list[str],
    *,
    triage_rows: list[dict] | None = None,
    target_pairs: int = 1000,
) -> dict:
    """Audit the pilot and report progress, yield and AI-human contingency."""
    audit = audit_review_notes(notes, baseline, pilot_ids)
    note_by_id = {row.get("source_question_id"): row for row in notes}
    pilot_rows = [note_by_id[qid] for qid in pilot_ids if qid in note_by_id]
    decisions = Counter(row.get("review_status") for row in pilot_rows)
    resolved = sum(decisions[state] for state in ("accepted", "rejected", "uncertain"))
    complete = (
        audit["status"] == "passed"
        and len(pilot_rows) == len(pilot_ids)
        and decisions["pending"] == 0
    )
    status = "invalid" if audit["status"] != "passed" else "complete" if complete else "in_progress"

    reviewers = Counter(
        str(row["reviewer"]).strip()
        for row in pilot_rows
        if row.get("review_status") != "pending" and isinstance(row.get("reviewer"), str)
    )
    actor_counts = Counter()
    for row in pilot_rows:
        if row.get("review_status") != "pending":
            try:
                actor_counts[reviewer_type(row)] += 1
            except ValueError:
                pass  # The audit already reports invalid actor types.
    accepted_rows = [row for row in pilot_rows if row.get("review_status") == "accepted"]
    error_types = Counter(row.get("error_type") for row in accepted_rows if row.get("error_type"))
    support_scores = Counter(
        str(row.get("explanation_support"))
        for row in accepted_rows
        if type(row.get("explanation_support")) is int
    )
    rejected_origins = Counter(_rejected_origin(row) for row in accepted_rows)

    acceptance = None
    natural_error_yield = None
    projection = None
    if complete:
        rate = decisions["accepted"] / len(pilot_ids)
        interval = wilson_interval(decisions["accepted"], len(pilot_ids))
        acceptance = {
            "accepted": decisions["accepted"],
            "total": len(pilot_ids),
            "rate": rate,
            "wilson_95": interval,
        }
        natural_count = rejected_origins["model"]
        natural_rate = natural_count / len(pilot_ids)
        natural_interval = wilson_interval(natural_count, len(pilot_ids))
        natural_error_yield = {
            "accepted": natural_count,
            "total": len(pilot_ids),
            "rate": natural_rate,
            "wilson_95": natural_interval,
            "scope": "accepted_pairs_with_unchanged_m1_rejected",
            "note": (
                "Yield under this review procedure, not M1's population error rate. "
                "Manual and synthetic negatives are excluded."
            ),
        }
        if natural_interval is not None:
            projection = _project_pool(target_pairs, natural_rate, natural_interval)
            if projection is not None:
                projection["scope"] = natural_error_yield["scope"]

    triage_by_id = {}
    triage_errors = []
    pilot_set = set(pilot_ids)
    for row in triage_rows or []:
        qid = row.get("source_question_id")
        verdict = row.get("ai_verdict")
        if qid in triage_by_id:
            triage_errors.append(f"duplicate AI triage ID: {qid}")
        elif qid not in pilot_set:
            triage_errors.append(f"AI triage ID outside pilot: {qid}")
        elif not isinstance(verdict, str) or verdict not in AI_TRIAGE_VERDICTS:
            triage_errors.append(f"AI triage verdict invalid: {qid}")
        else:
            triage_by_id[qid] = verdict
    contingency: dict[str, Counter] = {}
    for row in [] if triage_errors or audit["status"] != "passed" else pilot_rows:
        verdict = triage_by_id.get(row.get("source_question_id"))
        if verdict is None or row.get("review_status") == "pending":
            continue
        if reviewer_type(row) != "human":
            continue
        contingency.setdefault(verdict, Counter())[row["review_status"]] += 1

    full_counts = audit["counts"]
    full_review_ready = (
        audit["status"] == "passed"
        and full_counts.get("pending", 0) == 0
        and full_counts.get("accepted", 0) > 0
    )
    next_action = (
        "fix_invalid_reviews"
        if status == "invalid"
        else "finish_fixed_pilot"
        if status == "in_progress"
        else "review_pilot_yield_and_lock_expansion_protocol"
    )
    if triage_errors:
        status = "invalid"
        if audit["status"] == "passed":
            next_action = "fix_invalid_ai_triage"
    return {
        "schema": "preference_pilot_report_v1",
        "status": status,
        "pilot": {
            "records": len(pilot_ids),
            "resolved": resolved,
            "decisions": dict(sorted(decisions.items())),
            "reviewers": dict(sorted(reviewers.items())),
            "reviewer_types": dict(sorted(actor_counts.items())),
            "accepted_error_types": dict(sorted(error_types.items())),
            "accepted_support_scores": dict(sorted(support_scores.items())),
            "accepted_rejected_origins": dict(sorted(rejected_origins.items())),
        },
        "acceptance": acceptance,
        "natural_error_yield": natural_error_yield,
        "candidate_pool_projection": projection,
        "ai_triage": {
            "status": "invalid" if triage_errors else "valid" if triage_rows else "not_provided",
            "coverage": len(triage_by_id),
            "errors": triage_errors,
            "human_decision_contingency": {
                verdict: dict(sorted(counts.items()))
                for verdict, counts in sorted(contingency.items())
            },
            "note": (
                "The contingency is descriptive. AI verdicts and human decisions do not share "
                "the same label semantics, so no agreement score is claimed."
            ),
        },
        "audit": audit,
        "gates": {
            "pilot_complete": complete,
            "full_review_ready_for_export": full_review_ready,
            "accepted_subset_ready_for_export": (
                audit["status"] == "passed" and full_counts.get("accepted", 0) > 0
            ),
            "training_gate_open": False,
            "next_action": next_action,
        },
        "limitations": [
            "A complete pilot estimates review yield but does not validate unseen candidates.",
            "Overall acceptance includes constructed negatives; pool projection uses only "
            "accepted pairs with unchanged M1 negatives under the same review procedure.",
            (
                "The report checks metadata and consistency; it cannot verify visual truth "
                "or identity."
            ),
            "Training remains closed until accepted pairs are exported, validated and frozen.",
        ],
    }


def render_pilot_markdown(report: dict) -> str:
    """Render a compact, human-readable companion to the JSON report."""
    pilot = report["pilot"]
    decisions = pilot["decisions"]
    lines = [
        "# Báo cáo pilot dữ liệu preference",
        "",
        f"Trạng thái: **{report['status']}**.",
        "",
        "## Tiến độ duyệt",
        "",
        "| Tổng | Đã quyết định | Accepted | Rejected | Uncertain | Pending |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {pilot['records']} | {pilot['resolved']} | "
            f"{decisions.get('accepted', 0)} | {decisions.get('rejected', 0)} | "
            f"{decisions.get('uncertain', 0)} | {decisions.get('pending', 0)} |"
        ),
        "",
        "Nguồn quyết định: " + ", ".join(
            f"{actor}: {count}" for actor, count in pilot.get("reviewer_types", {}).items()
        ) + ". Nhãn AI không phải xác nhận bởi người.",
        "",
    ]
    acceptance = report.get("acceptance")
    if acceptance is None:
        lines.extend(
            [
                (
                    "Chưa tính tỷ lệ giữ lại hoặc quy mô pool cần thiết vì pilot cố định "
                    "chưa hoàn tất."
                ),
                "Không dùng trạng thái tạm thời này để mở train M2/M3.",
                "",
            ]
        )
    else:
        low, high = acceptance["wilson_95"]
        lines.extend(
            [
                "## Tỷ lệ giữ lại",
                "",
                (
                    f"Pilot giữ lại {acceptance['accepted']}/{acceptance['total']} mẫu "
                    f"({acceptance['rate']:.1%}); khoảng Wilson 95%: {low:.1%}–{high:.1%}."
                ),
                "",
            ]
        )
        projection = report.get("candidate_pool_projection")
        if projection is None:
            lines.append(
                "Không ước lượng pool lỗi tự nhiên: chưa có cặp accepted giữ nguyên lỗi M1."
            )
        else:
            lower, upper = projection["interval_candidates"]
            upper_text = str(upper) if upper is not None else "không xác định"
            lines.append(
                f"Để lập kế hoạch cho {projection['target_pairs']} cặp giữ nguyên lỗi M1: "
                "ước lượng điểm "
                f"{projection['point_estimate_candidates']} ứng viên; dải theo khoảng tin cậy "
                f"{lower}–{upper_text}. Đây là ước lượng, không phải quota phải đạt."
            )
        lines.append("")

        natural = report.get("natural_error_yield")
        if natural is not None:
            lines.extend(
                [
                    "Nguồn rejected của các cặp accepted: "
                    + ", ".join(
                        f"{kind}: {count}"
                        for kind, count in sorted(pilot["accepted_rejected_origins"].items())
                    )
                    + ".",
                    (
                        f"Cặp giữ nguyên lỗi M1: {natural['accepted']}/{natural['total']} "
                        f"({natural['rate']:.1%}). Chỉ nhóm này dùng để ước lượng pool; "
                        "không cộng negative do người/AI tạo. Đây là tỷ lệ thu được cặp "
                        "theo quy trình duyệt, không phải tỷ lệ M1 trả lời sai."
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Cổng tiếp theo",
            "",
            f"- Pilot hoàn tất: `{report['gates']['pilot_complete']}`.",
            f"- Dữ liệu đối chiếu AI: `{report['ai_triage']['status']}`.",
            (
                "- Toàn bộ review sẵn sàng export: "
                f"`{report['gates']['full_review_ready_for_export']}`."
            ),
            (
                "- Có subset accepted đủ metadata để export: "
                f"`{report['gates']['accepted_subset_ready_for_export']}`. "
                "Không đồng nghĩa đủ quy mô, đủ chất lượng hoặc đã đóng băng."
            ),
            "- Cổng train: `False` cho đến khi export, validate và freeze cặp accepted.",
            f"- Việc tiếp theo: `{report['gates']['next_action']}`.",
            "",
            "AI triage chỉ dùng để hỗ trợ và đối chiếu mô tả; không được tính là nhãn người.",
            "",
        ]
    )
    if report["ai_triage"]["errors"]:
        lines.extend(["Không xuất bảng đối chiếu AI do lỗi dữ liệu:", ""])
        lines.extend(f"- {error}" for error in report["ai_triage"]["errors"])
        lines.append("")
    return "\n".join(lines)
