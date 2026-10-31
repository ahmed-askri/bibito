from bibito.dagger import DAggerRollout, state_key, unseen_rows

ACTIONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]


def st(angle, dist=100.0, ammo=26.0):
    return {"ammo": ammo, "health": 100.0, "enemies": [{"name": "Demon", "distance": dist, "angle": angle}]}


def teacher(state):
    return "ATTACK"


def student(state):
    return "TURN_LEFT"


def test_state_key_ignores_dict_order():
    a = {"ammo": 1.0, "enemies": []}
    b = {"enemies": [], "ammo": 1.0}
    assert state_key(a) == state_key(b)


def test_unseen_rows_drops_exact_matches_only():
    ref = [{"state": st(10.0)}]
    rows = [{"state": st(10.0)}, {"state": st(10.1)}]
    out = unseen_rows(rows, ref)
    assert len(out) == 1 and out[0]["state"]["enemies"][0]["angle"] == 10.1


def test_rollout_labels_with_teacher_and_acts_with_student_when_no_noise():
    r = DAggerRollout(student, teacher, ACTIONS, seed0=100, round_id=1, noise=0.0)
    assert r(st(5.0)) == "TURN_LEFT"      # the student acts
    assert r.rows[0]["action"] == "ATTACK"  # the teacher labels
    assert r.rows[0]["round"] == 1 and r.rows[0]["seed"] == 100


def test_rollout_skips_states_without_enemies_but_still_acts():
    r = DAggerRollout(student, teacher, ACTIONS, seed0=0, round_id=1, noise=0.0)
    assert r({"ammo": 5.0, "enemies": []}) == "TURN_LEFT"
    assert r.rows == []


def test_rollout_noise_one_always_random_but_labels_stay_teacher():
    r = DAggerRollout(student, teacher, ACTIONS, seed0=0, round_id=1, noise=1.0, rng_seed=3)
    acts = {r(st(float(i))) for i in range(60)}
    assert acts <= set(ACTIONS) and len(acts) > 1
    assert all(row["action"] == "ATTACK" for row in r.rows)


def test_rollout_episode_counter_changes_seed_and_resets_step():
    r = DAggerRollout(student, teacher, ACTIONS, seed0=500, round_id=2, noise=0.0)
    r(st(1.0)); r(st(2.0))
    r.end_episode()
    r(st(3.0))
    assert [x["seed"] for x in r.rows] == [500, 500, 501]
    assert [x["step"] for x in r.rows] == [0, 1, 0]
