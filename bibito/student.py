import os

import torch
from torch import nn

from bibito.policies import ACTIONS

torch.set_num_threads(1)  # tiny model: threading only adds overhead and latency

ANGLE_SCALE, DIST_SCALE, AMMO_SCALE = 30.0, 400.0, 50.0


def featurize(state):
    """Nearest enemy's angle and distance, plus ammo, scaled to roughly [-1, 1]."""
    e = state["enemies"][0]
    return [e["angle"] / ANGLE_SCALE, e["distance"] / DIST_SCALE, state["ammo"] / AMMO_SCALE]


def rows_to_xy(rows):
    x = torch.tensor([featurize(r["state"]) for r in rows], dtype=torch.float32)
    y = torch.tensor([ACTIONS.index(r["action"]) for r in rows], dtype=torch.long)
    return x, y


class StudentNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def train(train_rows, val_rows, epochs=60, lr=1e-2, batch_size=128, seed=0, log=None):
    """Train on train_rows, keep the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    xt, yt = rows_to_xy(train_rows)
    xv, yv = rows_to_xy(val_rows)
    model = StudentNet()
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
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, state):
        if not state["enemies"]:
            return "ATTACK"  # same default as the scripted teacher
        x = torch.tensor([featurize(state)], dtype=torch.float32)
        with torch.no_grad():
            return ACTIONS[self.model(x).argmax(dim=1).item()]


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path):
    model = StudentNet()
    model.load_state_dict(torch.load(path))
    return model.eval()