import json
import os
import sys

import torch

from bibito.dataset import load_jsonl
from bibito.scenarios import get_scenario
from bibito.student import load_model, rows_to_xy

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
train = load_jsonl(f"data/{SCENARIO}_train.jsonl")
val = load_jsonl(f"data/{SCENARIO}_val.jsonl")


def key(row):
    return json.dumps(row["state"], sort_keys=True)


def game_fingerprints(rows):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(key(r))
    return {seed: hash(tuple(keys)) for seed, keys in by_seed.items()}


train_keys = {key(r) for r in train}
train_games = game_fingerprints(train)
val_games = game_fingerprints(val)

seen = [r for r in val if key(r) in train_keys]
unseen = [r for r in val if key(r) not in train_keys]

model, scenario = load_model(f"models/student_{SCENARIO}.pt")
actions = list(get_scenario(scenario).actions)


def acc(rows):
    if not rows:
        return float("nan")
    x, y = rows_to_xy(rows, actions)
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


lines = [
    f"train: {len(train_games)} games, {len(set(train_games.values()))} distinct; {len(train)} rows, {len(train_keys)} distinct states",
    f"val:   {len(val_games)} games, {len(set(val_games.values()))} distinct; {len(val)} rows",
    f"val games identical to a train game: {sum(h in set(train_games.values()) for h in val_games.values())}/{len(val_games)}",
    f"val rows whose exact state is in train: {len(seen)} ({len(seen) / len(val) * 100:.1f}%)",
    f"student agreement on SEEN val rows:   {acc(seen) * 100:.1f}% ({len(seen)} rows)",
    f"student agreement on UNSEEN val rows: {acc(unseen) * 100:.1f}% ({len(unseen)} rows)",
]
print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/overlap_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")