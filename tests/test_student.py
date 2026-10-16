import random

import pytest
import torch

from bibito.policies import make_scripted_policy
from bibito.scenarios import get_scenario
from bibito.student import (
    K,
    N_FEATURES,
    StudentNet,
    StudentPolicy,
    featurize,
    load_model,
    rows_to_xy,
    save_model,
    train,
)

ACTIONS = list(get_scenario("defend_the_center").actions)


def en(angle, distance=200.0):
    return {"name": "Demon", "distance": distance, "angle": angle}


def st(*enemies, ammo=50, health=100):
    return {"ammo": ammo, "health": health, "enemies": list(enemies)}


class Always(torch.nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape[0], 3)
        out[:, 1] = 5.0
        return out


def test_featurize_one_enemy_pads_the_rest_with_zeros():
    f = featurize(st(en(90.0, 200.0), ammo=25, health=50))
    assert len(f) == N_FEATURES
    assert f[:4] == pytest.approx([1.0, 0.5, 1.0, 0.5])
    assert f[4:12] == [0.0] * 8
    assert f[12:] == pytest.approx([0.5, 0.5])


def test_featurize_small_angles_are_stretched_and_health_defaults():
    f = featurize({"ammo": 50, "enemies": [en(-3.0)]})
    assert f[2] == pytest.approx(-0.2)
    assert f[-1] == pytest.approx(1.0)


def test_featurize_keeps_only_the_k_nearest():
    enemies = [en(10.0, 100.0 + i) for i in range(K + 2)]
    assert len(featurize(st(*enemies))) == N_FEATURES


def test_rows_to_xy_maps_actions_to_indices():
    rows = [{"state": st(en(10)), "action": "TURN_RIGHT"}, {"state": st(en(0)), "action": "ATTACK"}]
    x, y = rows_to_xy(rows, ACTIONS)
    assert x.shape == (2, N_FEATURES)
    assert y.tolist() == [1, 2]


def test_policy_follows_the_model_and_defaults_when_no_enemy():
    policy = StudentPolicy(Always(), "defend_the_center")
    assert policy(st(en(3.0))) == "TURN_RIGHT"
    assert policy({"ammo": 50, "enemies": []}) == "ATTACK"


def synthetic_rows(n, seed):
    rng = random.Random(seed)
    teacher = make_scripted_policy(5, "defend_the_center")
    rows = []
    for _ in range(n):
        enemies = sorted(
            [en(rng.uniform(-180, 180), rng.uniform(100, 500)) for _ in range(rng.randint(1, 3))],
            key=lambda e: e["distance"],
        )
        s = st(*enemies)
        rows.append({"state": s, "action": teacher(s)})
    return rows


def test_training_learns_the_teacher_rule_on_synthetic_data():
    model, val_acc = train(synthetic_rows(3000, 1), synthetic_rows(500, 2), ACTIONS, epochs=60)
    assert val_acc >= 0.85


def test_save_and_load_round_trip(tmp_path):
    model = StudentNet(3)
    path = str(tmp_path / "m.pt")
    save_model(model, path, "defend_the_center")
    loaded, scenario = load_model(path)
    assert scenario == "defend_the_center"
    x = torch.rand(4, N_FEATURES)
    assert torch.allclose(model.eval()(x), loaded(x))