"""Deterministic lexical explanation metric; visual grounding needs separate review."""

from .answer_metrics import normalize_answer


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return token-level longest common subsequence length with linear memory."""
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, reference_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if token == reference_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, references: list[str]) -> float:
    """Score against the best reference with NFC, lowercase, punctuation and space normalization.

    This measures word-order overlap, not whether the explanation is true in the image.
    Empty predictions score zero; references must contain at least one nonempty string.
    """
    if not isinstance(prediction, str):
        raise ValueError("Prediction explanation must be a string")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(reference, str) or not reference.strip() for reference in references)
    ):
        raise ValueError("Reference explanations must be a nonempty list of nonempty strings")
    reference_tokens = [normalize_answer(reference).split() for reference in references]
    if any(not tokens for tokens in reference_tokens):
        raise ValueError("Reference explanation is empty after normalization")
    predicted_tokens = normalize_answer(prediction).split()
    if not predicted_tokens:
        return 0.0
    scores = []
    for tokens in reference_tokens:
        overlap = _lcs_length(predicted_tokens, tokens)
        scores.append(2 * overlap / (len(predicted_tokens) + len(tokens)))
    return max(scores)
