import os
import sys

from bibito.dataset import Recorder, action_counts, assert_disjoint, majority_baseline, save_jsonl
from bibito.evaluate import run_episodes
from bibito.policies import make_scripted_policy, pick_target

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TRAIN_SEED0 = 10000
VAL_SEED0, N_VAL = 2000, 100
THRESHOLD = 5


def record(seed0, n_episodes):
    rec = Recorder(make_scripted_policy(THRESHOLD, SCENARIO), seed0)

    def log(done, total, ep):
        rec.end_episode()
        if done % 20 == 0:
            print(f">> {done}/{total}", flush=True)

    run_episodes(rec, n_episodes=n_episodes, seed0=seed0, scenario=SCENARIO, log=log)
    return rec.rows


def near_threshold(row):
    return abs(abs(pick_target(row["state"]["enemies"])["angle"]) - THRESHOLD) <= 2


def contested(row):
    es = row["state"]["enemies"]
    return sum(e["distance"] <= es[0]["distance"] * 1.15 for e in es) > 1


def report(name, rows):
    counts = action_counts(rows)
    total = len(rows)
    episodes = len({r["seed"] for r in rows})
    near = sum(near_threshold(r) for r in rows)
    cont = sum(contested(r) for r in rows)
    return [
        f"{name}: {episodes} episodes, {total} rows",
        "  actions: " + ", ".join(f"{a} {c} ({c / total * 100:.0f}%)" for a, c in sorted(counts.items())),
        f"  majority-class baseline: {majority_baseline(rows) * 100:.0f}%",
        f"  rows near the threshold (target within {THRESHOLD}+-2 degrees): {near} ({near / total * 100:.0f}%)",
        f"  rows with several enemies at similar distance (contested): {cont} ({cont / total * 100:.0f}%)",
    ]


train = record(TRAIN_SEED0, N_TRAIN)
val = record(VAL_SEED0, N_VAL)
assert_disjoint(train, val)
save_jsonl(train, f"data/{SCENARIO}_train.jsonl")
save_jsonl(val, f"data/{SCENARIO}_val.jsonl")

lines = [f"scenario: {SCENARIO}, teacher: scripted t{THRESHOLD}"] + report("train", train) + report("val", val)
print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/dataset_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")