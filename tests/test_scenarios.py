import pytest
from bibito.state import build_state
from bibito.scenarios import SCENARIOS, get_scenario
CACO = {"Cacodemon"}

def test_every_scenario_has_left_right_attack_in_its_actions():
    for sc in SCENARIOS.values():
        assert sc.left in sc.actions
        assert sc.right in sc.actions
        assert sc.attack in sc.actions
        assert len(set(sc.actions)) == len(sc.actions)


def test_defend_the_center_uses_turning_and_two_enemy_types():
    sc = get_scenario("defend_the_center")
    assert sc.actions == ("TURN_LEFT", "TURN_RIGHT", "ATTACK")
    assert sc.enemies == frozenset({"Demon", "MarineChainsawVzd"})


def test_unknown_scenario_lists_the_known_ones():
    with pytest.raises(KeyError, match="basic"):
        get_scenario("nope")

def test_health_is_included_only_when_given():
    with_health = build_state(26, 0, 0, 0, [], CACO, health=80)
    without = build_state(26, 0, 0, 0, [], CACO)
    assert with_health["health"] == 80
    assert "health" not in without


def test_enemy_behind_is_180_degrees():
    s = build_state(26, 0, 0, 0, [("Cacodemon", -100, 0)], CACO)
    assert abs(s["enemies"][0]["angle"]) == pytest.approx(180.0)


def test_player_angle_wraparound():
    # player faces 350 degrees, enemy straight along +x: it is 10 degrees to the left
    s = build_state(26, 0, 0, 350, [("Cacodemon", 100, 0)], CACO)
    assert s["enemies"][0]["angle"] == pytest.approx(10.0)


def test_multiple_enemy_types_are_kept_and_sorted_by_distance():
    names = {"Demon", "MarineChainsawVzd"}
    objs = [("Demon", 500, 0), ("MarineChainsawVzd", 0, 200), ("DoomPlayer", 0, 0), ("BulletPuff", 5, 5)]
    s = build_state(26, 0, 0, 0, objs, names)
    assert [e["name"] for e in s["enemies"]] == ["MarineChainsawVzd", "Demon"]