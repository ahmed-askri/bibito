import json
import os
from collections import Counter


class Recorder:
    """Wraps a policy and logs every (state, action) it produces.

    States with no enemies are not recorded: the teacher's answer there is an
    arbitrary default, so those rows would be noise.
    """

    def __init__(self, policy, seed0):
        self.policy = policy
        self.seed0 = seed0
        self.episode = 0
        self.step = 0
        self.rows = []

    def __call__(self, state):
        action = self.policy(state)
        if state["enemies"]:
            self.rows.append({
                "seed": self.seed0 + self.episode,
                "step": self.step,
                "state": state,
                "action": action,
            })
        self.step += 1
        return action

    def end_episode(self):
        self.episode += 1
        self.step = 0


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_counts(rows):
    return Counter(r["action"] for r in rows)


def majority_baseline(rows):
    """Accuracy of always predicting the most common action: the number a student must beat."""
    return max(action_counts(rows).values()) / len(rows)


def assert_disjoint(rows_a, rows_b):
    """Fail loudly if the same game (seed) appears in both sets."""
    overlap = {r["seed"] for r in rows_a} & {r["seed"] for r in rows_b}
    if overlap:
        raise ValueError(f"seeds appear in both sets: {sorted(overlap)[:5]}")