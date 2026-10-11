from bibito.policies import ACTIONS, always_attack, make_random_policy, make_scripted_policy


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