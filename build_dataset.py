import os
import sys

from bibito.dataset import Recorder, action_counts, assert_disjoint, majority_baseline, save_jsonl
from bibito.evaluate import run_episodes
from bibito.policies import make_scripted_policy

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TRAIN_SEED0 = 10000
VAL_SEED0, N_VAL = 2000, 100


def record(seed0, n_episodes):
    rec = Recorder(make_scripted_policy(5), seed0)
    run_episodes(rec, n_episodes=n_episodes, seed0=seed0,
                 log=lambda done, total, ep: rec.end_episode())
    return rec.rows


def report(name, rows):
    counts = action_counts(rows)
    total = len(rows)
    episodes = len({r["seed"] for r in rows})
    boundary = sum(4 <= abs(r["state"]["enemies"][0]["angle"]) <= 6 for r in rows)
    return [
        f"{name}: {episodes} episodes, {total} rows",
        "  actions: " + ", ".join(f"{a} {c} ({c / total * 100:.0f}%)" for a, c in sorted(counts.items())),
        f"  majority-class baseline: {majority_baseline(rows) * 100:.0f}%",
        f"  rows near the teacher's threshold (4 <= |angle| <= 6): {boundary} ({boundary / total * 100:.0f}%)",
    ]


train = record(TRAIN_SEED0, N_TRAIN)
val = record(VAL_SEED0, N_VAL)
assert_disjoint(train, val)
save_jsonl(train, "data/train.jsonl")
save_jsonl(val, "data/val.jsonl")

lines = report("train", train) + report("val", val)
print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open("results/dataset.txt", "w") as f:
    f.write("\n".join(lines) + "\n")