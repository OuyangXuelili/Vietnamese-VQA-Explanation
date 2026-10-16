from datetime import timedelta

import pytest

from vqa_explanation.preference.audit import _valid_time
from vqa_explanation.utils.review_time import parse_review_time


@pytest.mark.parametrize("timestamp", [
    "2026-10-12T03:00:00.000Z", "2026-10-12T03:00:00+00:00",
    "2026-10-12T10:00:00+07:00",
])
def test_browser_and_offset_times_represent_same_instant(timestamp):
    actual = parse_review_time(timestamp)
    expected = parse_review_time("2026-10-12T03:00:00.000Z")
    assert actual == expected
    assert _valid_time(timestamp)
    assert expected.utcoffset() == timedelta(0)


@pytest.mark.parametrize("timestamp", ["", "2026-10-12", "bad", "2026-10-12T03:00:00"])
def test_missing_or_invalid_timezone_is_rejected(timestamp):
    with pytest.raises(ValueError):
        parse_review_time(timestamp)
    assert not _valid_time(timestamp)
