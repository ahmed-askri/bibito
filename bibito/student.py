import os

import torch
from torch import nn

from bibito.scenarios import get_scenario

torch.set_num_threads(1)  # tiny model: threading only adds overhead and latency

K = 3  # how many of the nearest enemies the student sees
ANGLE_SCALE = 180.0  # full range of the angle
NEAR_SCALE = 15.0    # second angle feature, clipped: stretches the region around "straight ahead"
DIST_SCALE, AMMO_SCALE, HEALTH_SCALE = 400.0, 50.0, 100.0
N_FEATURES = 4 * K + 2


def featurize(state):
    """Per enemy (nearest first): present, angle, clipped fine angle, distance. Then ammo and health."""
    enemies = state["enemies"][:K]
    f = []
    for i in range(K):
        if i < len(enemies):
            e = enemies[i]
            f += [1.0, e["angle"] / ANGLE_SCALE,
                  max(-1.0, min(1.0, e["angle"] / NEAR_SCALE)), e["distance"] / DIST_SCALE]
        else:
            f += [0.0, 0.0, 0.0, 0.0]
    f += [state["ammo"] / AMMO_SCALE, state.get("health", HEALTH_SCALE) / HEALTH_SCALE]
    return f


def rows_to_xy(rows, actions):
    x = torch.tensor([featurize(r["state"]) for r in rows], dtype=torch.float32)
    y = torch.tensor([actions.index(r["action"]) for r in rows], dtype=torch.long)
    return x, y


class StudentNet(nn.Module):
    def __init__(self, n_actions, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def train(train_rows, val_rows, actions, epochs=40, lr=3e-3, batch_size=256, seed=0, log=None):
    """Train on train_rows, keep the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    xt, yt = rows_to_xy(train_rows, actions)
    xv, yv = rows_to_xy(val_rows, actions)
    model = StudentNet(len(actions))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        val_acc = accuracy(model, xv, yv)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if log:
            log(epoch, loss.item(), val_acc)

    model.load_state_dict(best_state)
    return model, best_acc


class StudentPolicy:
    def __init__(self, model, scenario="basic"):
        self.model = model.eval()
        self.sc = get_scenario(scenario)
        self.actions = list(self.sc.actions)

    def __call__(self, state):
        if not state["enemies"]:
            return self.sc.attack  # same default as the scripted teacher
        x = torch.tensor([featurize(state)], dtype=torch.float32)
        with torch.no_grad():
            return self.actions[self.model(x).argmax(dim=1).item()]


def save_model(model, path, scenario):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"scenario": scenario, "hidden": model.hidden,
                "n_actions": len(get_scenario(scenario).actions),
                "state_dict": model.state_dict()}, path)


def load_model(path):
    blob = torch.load(path)
    model = StudentNet(blob["n_actions"], blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["scenario"]