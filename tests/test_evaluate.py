import pytest

from bibito.evaluate import percentile, summarize


def test_percentile_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10


def test_summarize():
    eps = [
        {"reward": 100, "steps": 20, "killed": True, "latencies_ms": [1.0, 2.0]},
        {"reward": 100, "steps": 40, "killed": True, "latencies_ms": [3.0]},
        {"reward": -300, "steps": 300, "killed": False, "latencies_ms": [4.0]},
    ]
    s = summarize(eps)
    assert s["episodes"] == 3
    assert s["reward_mean"] == pytest.approx(-100 / 3)
    assert s["kill_rate"] == pytest.approx(2 / 3)
    assert s["steps_mean"] == pytest.approx(120)
    assert s["latency_p50_ms"] == 2.0

def test_summarize_death_rate():
    eps = [
        {"reward": 3, "steps": 400, "killed": True, "died": True, "latencies_ms": [1.0]},
        {"reward": 5, "steps": 2100, "killed": False, "died": False, "latencies_ms": [1.0]},
    ]
    assert summarize(eps)["death_rate"] == pytest.approx(0.5)