import numpy as np
import pytest

from scripts.benchmark_sft_collator import compare


def test_benchmark_reports_equality_but_rejects_changed_labels():
    rows = [{"question_id": "train-1"}]

    def reference(records):
        return {"input_ids": np.array([[1, 2]]), "labels": np.array([[-100, 2]])}

    report = compare(rows, reference, reference, repeats=1)
    assert report["equal_samples"] == 1
    assert "not GPU training" in report["note"]

    def wrong(records):
        return {"input_ids": np.array([[1, 2]]), "labels": np.array([[1, 2]])}

    with pytest.raises(AssertionError):
        compare(rows, reference, wrong, repeats=1)
    with pytest.raises(ValueError):
        compare([], reference, reference)
    with pytest.raises(ValueError):
        compare(rows, reference, reference, repeats=0)
