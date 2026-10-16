from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ERROR_TYPES = {"object", "attribute", "relation_action", "natural_model_error"}
REVIEW_STATES = {"pending", "accepted", "rejected", "uncertain"}
AI_TRIAGE_VERDICTS = {
    "pass_grounded",
    "repair_explanation",
    "exclude_answer_mismatch",
    "exclude_ambiguous",
    "exclude_out_of_scope",
}


def reviewer_type(row: dict) -> str:
    """Legacy reviews are human; AI decisions must declare their actor type."""
    value = row.get("reviewer_type")
    if value is None or value == "":
        return "human"
    if not isinstance(value, str) or value not in {"human", "ai"}:
        raise ValueError("reviewer_type must be human or ai")
    return value


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    source_question_id: str
    image_path: str
    prompt: str
    chosen: str
    rejected: str
    error_type: str
    generation_method: str
    review_status: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PreferencePair":
        if not isinstance(raw, dict):
            raise TypeError("preference pair must be a JSON object")
        required = [
            "pair_id",
            "source_question_id",
            "image_path",
            "prompt",
            "chosen",
            "rejected",
            "error_type",
            "generation_method",
            "review_status",
        ]
        missing = [key for key in required if key not in raw or raw[key] is None]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        invalid = [key for key in required if not isinstance(raw[key], str) or not raw[key].strip()]
        if invalid:
            raise ValueError(f"fields must be nonempty strings: {invalid}")
        if raw["chosen"].strip() == raw["rejected"].strip():
            raise ValueError("chosen and rejected must be different")
        if raw["error_type"] not in ERROR_TYPES:
            raise ValueError(f"unsupported error_type: {raw['error_type']}")
        if raw["review_status"] not in REVIEW_STATES:
            raise ValueError(f"unsupported review_status: {raw['review_status']}")
        return cls(**{key: raw[key].strip() for key in required})
