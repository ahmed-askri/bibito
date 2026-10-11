import pytest
from bibito.state import build_state, make_actions, normalize_angle

CACO = {"Cacodemon"}


def test_enemy_ahead_left_is_positive_45():
    s = build_state(50, 0, 0, 0, [("Cacodemon", 100, 100)], CACO)
    e = s["enemies"][0]
    assert e["angle"] == pytest.approx(45.0)
    assert e["distance"] == pytest.approx(141.4, abs=0.1)


def test_angle_is_relative_to_facing_direction():
    s = build_state(50, 0, 0, 90, [("Cacodemon", 0, 100)], CACO)
    assert s["enemies"][0]["angle"] == pytest.approx(0.0)


def test_normalize_angle():
    assert normalize_angle(350) == pytest.approx(-10)
    assert normalize_angle(-190) == pytest.approx(170)


def test_player_dead_enemies_and_decor_are_ignored():
    objs = [("DoomPlayer", 0, 0), ("DeadCacodemon", 50, 0), ("Blood", 10, 10)]
    assert build_state(50, 0, 0, 0, objs, CACO)["enemies"] == []


def test_enemies_sorted_by_distance():
    objs = [("Cacodemon", 300, 0), ("Cacodemon", 100, 0)]
    d = [e["distance"] for e in build_state(50, 0, 0, 0, objs, CACO)["enemies"]]
    assert d == sorted(d)


def test_make_actions_one_hot():
    names, vec = make_actions(["A", "B", "C"])
    assert names == ["A", "B", "C"]
    assert vec == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]