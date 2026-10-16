import math

import pytest

from bibito.router import escalation_curve, least_confident


def test_least_confident_returns_the_lowest_confidence_indices():
    assert least_confident([0.9, 0.2, 0.5, 0.1], 2) == [3, 1]


def test_escalating_nothing_gives_student_accuracy():
    rows = escalation_curve([0.9, 0.4], [True, False], {}, ks=[0])
    assert rows[0]["router_acc"] == pytest.approx(0.5)
    assert rows[0]["threshold"] is None
    assert math.isnan(rows[0]["llm_acc_on_escalated"])


def test_escalating_wrong_rows_to_a_correct_llm_raises_accuracy():
    conf = [0.9, 0.8, 0.4, 0.3]
    student_correct = [True, True, False, False]
    rows = escalation_curve(conf, student_correct, {2: True, 3: True}, ks=[0, 2])
    assert rows[0]["router_acc"] == pytest.approx(0.5)
    assert rows[1]["router_acc"] == pytest.approx(1.0)
    assert rows[1]["escalated"] == pytest.approx(0.5)
    assert rows[1]["threshold"] == pytest.approx(0.4)
    assert rows[1]["student_acc_on_escalated"] == pytest.approx(0.0)
    assert rows[1]["llm_acc_on_escalated"] == pytest.approx(1.0)


def test_escalating_to_a_worse_llm_lowers_accuracy():
    rows = escalation_curve([0.9, 0.4], [True, True], {1: False}, ks=[1])
    assert rows[0]["router_acc"] == pytest.approx(0.5)


def test_missing_llm_answer_raises():
    with pytest.raises(KeyError):
        escalation_curve([0.9, 0.4], [True, True], {}, ks=[1])