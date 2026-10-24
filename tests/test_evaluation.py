import numpy as np
import pytest

import evaluation as ev


def test_confusion_and_formulas():
    actual = [1, 1, 1, 0, 0, 0, 0, 1]
    pred = [1, 1, 0, 0, 0, 1, 0, 1]
    c = ev.confusion_counts(actual, pred)
    assert c == dict(TP=3, TN=3, FP=1, FN=1)
    assert ev.accuracy(c) == pytest.approx(6 / 8)
    assert ev.precision(c) == pytest.approx(3 / 4)
    assert ev.recall(c) == pytest.approx(3 / 4)
    assert ev.specificity(c) == pytest.approx(3 / 4)
    assert ev.false_positive_rate(c) == pytest.approx(1 / 4)


def test_zero_denominators_are_none_and_shown_as_na():
    c = ev.confusion_counts([0, 0], [0, 0])
    assert ev.precision(c) is None and ev.recall(c) is None
    assert ev.specificity(c) == 1.0 and ev.false_positive_rate(c) == 0.0
    assert ev.accuracy(dict(TP=0, TN=0, FP=0, FN=0)) is None
    assert ev.fmt(None) == "N/A" and ev.fmt(float("nan")) == "N/A"


def test_mae_and_improvement():
    assert ev.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert ev.mae([1, np.nan], [2, 5]) == pytest.approx(1.0)
    assert ev.mae([np.nan], [1]) is None
    assert ev.improvement_pct(4.0, 1.0) == pytest.approx(75.0)
    assert ev.improvement_pct(1.0, 2.0) == pytest.approx(-100.0)
    assert ev.improvement_pct(0.0, 1.0) is None
    assert ev.improvement_pct(None, 1.0) is None


def test_detection_delay():
    assert ev.detection_delay(95, 80) == 15
    assert ev.detection_delay(None, 80) is None          # missed: never invented
    assert ev.detection_delay(70, 80) is None            # before the fault: not a valid detection


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ev.confusion_counts([1, 0], [1])
