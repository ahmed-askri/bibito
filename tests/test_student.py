import random

import pytest
import torch

from bibito.policies import ACTIONS, make_scripted_policy
from bibito.student import StudentPolicy, featurize, rows_to_xy, train


def st(angle, distance=385.0, ammo=50):
    return {"ammo": ammo, "enemies": [{"name": "Cacodemon", "distance": distance, "angle": angle}]}


def test_featurize_scales_inputs():
    assert featurize(st(15.0, 200.0, 25)) == pytest.approx([0.5, 0.5, 0.5])


def test_rows_to_xy_maps_actions_to_indices():
    rows = [{"state": st(10), "action": "MOVE_RIGHT"}, {"state": st(0), "action": "ATTACK"}]
    x, y = rows_to_xy(rows)
    assert x.shape == (2, 3)
    assert y.tolist() == [ACTIONS.index("MOVE_RIGHT"), ACTIONS.index("ATTACK")]


def test_policy_follows_the_model_and_defaults_when_no_enemy():
    class Always(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], len(ACTIONS))
            out[:, 1] = 5.0
            return out

    policy = StudentPolicy(Always())
    assert policy(st(3.0)) == ACTIONS[1]
    assert policy({"ammo": 50, "enemies": []}) == "ATTACK"


def synthetic_rows(n, seed):
    rng = random.Random(seed)
    teacher = make_scripted_policy(5)
    rows = []
    for _ in range(n):
        s = st(rng.uniform(-25, 25), rng.uniform(300, 400), 50)
        rows.append({"state": s, "action": teacher(s)})
    return rows


def test_training_learns_the_teacher_rule_on_synthetic_data():
    model, val_acc = train(synthetic_rows(1500, 1), synthetic_rows(300, 2), epochs=60)
    assert val_acc >= 0.95