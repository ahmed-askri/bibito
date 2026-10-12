import pytest

from bibito.dataset import (
    Recorder,
    action_counts,
    assert_disjoint,
    load_jsonl,
    majority_baseline,
    save_jsonl,
)

S = {"ammo": 50, "enemies": [{"name": "Cacodemon", "distance": 385.0, "angle": 12.0}]}
EMPTY = {"ammo": 49, "enemies": []}


def test_recorder_logs_rows_and_passes_action_through():
    rec = Recorder(lambda s: "MOVE_LEFT", seed0=10000)
    assert rec(S) == "MOVE_LEFT"
    assert rec.rows == [{"seed": 10000, "step": 0, "state": S, "action": "MOVE_LEFT"}]


def test_end_episode_moves_to_next_seed_and_resets_step():
    rec = Recorder(lambda s: "ATTACK", seed0=10000)
    rec(S)
    rec(S)
    rec.end_episode()
    rec(S)
    assert [(r["seed"], r["step"]) for r in rec.rows] == [(10000, 0), (10000, 1), (10001, 0)]


def test_states_without_enemies_are_not_recorded():
    rec = Recorder(lambda s: "ATTACK", seed0=1)
    assert rec(EMPTY) == "ATTACK"  # the game still gets an action
    assert rec.rows == []


def test_jsonl_round_trip(tmp_path):
    rows = [{"seed": 1, "step": 0, "state": S, "action": "ATTACK"}]
    path = str(tmp_path / "sub" / "rows.jsonl")
    save_jsonl(rows, path)
    assert load_jsonl(path) == rows


def test_action_counts_and_majority_baseline():
    rows = [{"action": "ATTACK"}] * 3 + [{"action": "MOVE_LEFT"}]
    assert dict(action_counts(rows)) == {"ATTACK": 3, "MOVE_LEFT": 1}
    assert majority_baseline(rows) == pytest.approx(0.75)


def test_assert_disjoint():
    a = [{"seed": 1}, {"seed": 2}]
    assert_disjoint(a, [{"seed": 3}])
    with pytest.raises(ValueError):
        assert_disjoint(a, [{"seed": 2}])