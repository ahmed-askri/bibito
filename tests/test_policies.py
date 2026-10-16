from bibito.policies import ACTIONS, always_attack, make_random_policy, make_scripted_policy
from bibito.policies import pick_target

def st(*angles):
    return {"ammo": 50, "enemies": [{"name": "Cacodemon", "distance": 300, "angle": a} for a in angles]}


def test_strafes_left_when_enemy_is_left():
    assert make_scripted_policy(3)(st(10)) == "MOVE_LEFT"


def test_strafes_right_when_enemy_is_right():
    assert make_scripted_policy(3)(st(-10)) == "MOVE_RIGHT"


def test_attacks_when_aligned():
    p = make_scripted_policy(3)
    assert p(st(2)) == "ATTACK"
    assert p(st(-2)) == "ATTACK"
    assert p(st(3.0)) == "ATTACK"  # boundary is inclusive


def test_uses_nearest_enemy_first_in_list():
    assert make_scripted_policy(3)(st(1, 40)) == "ATTACK"


def test_no_enemies_returns_valid_action():
    assert make_scripted_policy(3)({"ammo": 50, "enemies": []}) in ACTIONS


def test_random_policy_is_valid_and_reproducible():
    a, b = make_random_policy(7), make_random_policy(7)
    seq_a = [a(st()) for _ in range(20)]
    seq_b = [b(st()) for _ in range(20)]
    assert seq_a == seq_b
    assert all(x in ACTIONS for x in seq_a)


def test_always_attack():
    assert always_attack(st(30)) == "ATTACK"

D = "defend_the_center"


def enemy(distance, angle):
    return {"name": "Demon", "distance": distance, "angle": angle}


def state_of(*enemies):
    return {"ammo": 26, "health": 100, "enemies": list(enemies)}


def test_defend_uses_turn_actions():
    p = make_scripted_policy(5, scenario=D)
    assert p(state_of(enemy(300, 30))) == "TURN_LEFT"
    assert p(state_of(enemy(300, -30))) == "TURN_RIGHT"
    assert p(state_of(enemy(300, 1))) == "ATTACK"


def test_defend_prefers_least_turning_among_equally_near_enemies():
    # nearest is 300 away at 90 degrees; the one at 310 needs only 4 degrees of turning
    p = make_scripted_policy(3, scenario=D)
    assert p(state_of(enemy(300, 90), enemy(310, 4), enemy(600, 0))) == "TURN_LEFT"
    assert p(state_of(enemy(300, 90), enemy(310, 2), enemy(600, 0))) == "ATTACK"


def test_defend_much_closer_enemy_takes_priority():
    p = make_scripted_policy(5, scenario=D)
    assert p(state_of(enemy(55, 40), enemy(700, 0))) == "TURN_LEFT"


def test_defend_no_enemies_returns_attack():
    assert make_scripted_policy(5, scenario=D)(state_of()) == "ATTACK"


def test_random_policy_uses_the_scenario_actions():
    p = make_random_policy(0, scenario=D)
    assert {p(state_of()) for _ in range(60)} == {"TURN_LEFT", "TURN_RIGHT", "ATTACK"}




def test_pick_target_none_without_enemies():
    assert pick_target([]) is None


def test_pick_target_prefers_smaller_angle_among_similar_distances():
    near = {"name": "Demon", "distance": 100.0, "angle": 40.0}
    similar = {"name": "Demon", "distance": 110.0, "angle": -10.0}
    assert pick_target([near, similar]) is similar


def test_pick_target_ignores_much_farther_enemy():
    near = {"name": "Demon", "distance": 100.0, "angle": 40.0}
    far = {"name": "Demon", "distance": 300.0, "angle": 1.0}
    assert pick_target([near, far]) is near