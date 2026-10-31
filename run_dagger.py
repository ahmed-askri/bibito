"""Phase 10: DAgger.

The student plays (with 10% random actions), the scripted teacher labels the states it visits,
those rows are added to the training data, the student is retrained from scratch. Repeat.
At the end, every round is evaluated on the same validation games as the original student.

Usage (from the project root, after build_dataset.py and train_student.py):
    python -u run_dagger.py [scenario] [rounds=3] [episodes_per_round=100] [n_eval=100]
"""
import os
import sys

from bibito.dagger import DAggerRollout, unseen_rows
from bibito.dataset import load_jsonl, save_jsonl
from bibito.evaluate import FRAME_SKIP, run_episodes, summarize
from bibito.harness import bootstrap_ci, distinct_outcomes, file_sha256, paired_diff
from bibito.policies import make_scripted_policy
from bibito.scenarios import get_scenario
from bibito.student import StudentPolicy, accuracy, load_model, rows_to_xy, save_model, train

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EPISODES_PER_ROUND = int(sys.argv[3]) if len(sys.argv) > 3 else 100
N_EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 100
DAGGER_SEED0 = 20000   # new seeds: not in train (10000+), validation (2000-2099) or test (9000-9099)
EVAL_SEED0 = 2000      # same validation games for every model
NOISE = 0.1
THRESHOLD = 5

paths = {
    "train": f"data/{SCENARIO}_train.jsonl",
    "val": f"data/{SCENARIO}_val.jsonl",
    "model": f"models/student_{SCENARIO}.pt",
}
missing = [p for p in paths.values() if not os.path.exists(p)]
if missing:
    raise SystemExit(f"missing {missing}. Run first: python -u build_dataset.py {SCENARIO} 200 "
                     f"and python -u train_student.py {SCENARIO}")

actions = list(get_scenario(SCENARIO).actions)
teacher = make_scripted_policy(THRESHOLD, SCENARIO)
train_rows = load_jsonl(paths["train"])
val_rows = load_jsonl(paths["val"])
base_model, _ = load_model(paths["model"])

# Honest validation: only rows whose exact state is not in the ORIGINAL training data.
unseen_val = unseen_rows(val_rows, train_rows)
x_unseen, y_unseen = rows_to_xy(unseen_val, actions)

lines = [f"scenario: {SCENARIO}, DAgger: {ROUNDS} rounds x {EPISODES_PER_ROUND} games, noise {NOISE}, "
         f"frame_skip {FRAME_SKIP}, rollout seeds {DAGGER_SEED0}+, eval seeds {EVAL_SEED0}..{EVAL_SEED0 + N_EVAL - 1}",
         f"original train rows {len(train_rows)}, validation rows {len(val_rows)}, "
         f"validation rows with unseen state {len(unseen_val)}"]


def say(text):
    lines.append(text)
    print(text, flush=True)


print(lines[0], flush=True)
print(lines[1], flush=True)

# ---- DAgger loop ----
models = [("round 0 (teacher data only)", base_model)]
data = list(train_rows)
model = base_model
for r in range(1, ROUNDS + 1):
    seed0 = DAGGER_SEED0 + (r - 1) * EPISODES_PER_ROUND
    rollout = DAggerRollout(StudentPolicy(model, SCENARIO), teacher, actions, seed0, r, noise=NOISE, rng_seed=r)

    def log(done, total, ep):
        rollout.end_episode()
        if done % 20 == 0:
            print(f">> round {r}: {done}/{total} games", flush=True)

    episodes = run_episodes(rollout, n_episodes=EPISODES_PER_ROUND, seed0=seed0, scenario=SCENARIO, log=log)
    new_rows = rollout.rows

    # How often did the current student disagree with the teacher on the states it actually visited?
    x_new, y_new = rows_to_xy(new_rows, actions)
    visited_agreement = accuracy(model, x_new, y_new)
    data += new_rows
    say(f"round {r}: {len(new_rows)} new rows from {EPISODES_PER_ROUND} student games "
        f"(mean kills while playing with noise {sum(e['reward'] for e in episodes) / len(episodes):.1f}); "
        f"student agreement with teacher on the states it visited: {visited_agreement * 100:.1f}%")

    model, best = train(data, val_rows, actions)  # retrain from scratch on all data so far
    models.append((f"round {r} (+{len(data) - len(train_rows)} DAgger rows)", model))

os.makedirs("models", exist_ok=True)
final_path = f"models/student_{SCENARIO}_dagger.pt"
save_model(model, final_path, SCENARIO)
save_jsonl(data, f"data/{SCENARIO}_dagger_train.jsonl")

# ---- evaluation: same validation games for everyone ----
say("")
say(f"evaluation on validation seeds {EVAL_SEED0}..{EVAL_SEED0 + N_EVAL - 1} ({N_EVAL} games each)")
teacher_eps = run_episodes(teacher, n_episodes=N_EVAL, seed0=EVAL_SEED0, scenario=SCENARIO)
teacher_kills = [e["reward"] for e in teacher_eps]
lo, hi = bootstrap_ci(teacher_kills)
say(f"{'teacher t5':<34} kills {sum(teacher_kills) / N_EVAL:.1f} (95% CI {lo:.1f}..{hi:.1f})")

baseline_kills = None
for name, m in models:
    agreement = accuracy(m, x_unseen, y_unseen)
    eps = run_episodes(StudentPolicy(m, SCENARIO), n_episodes=N_EVAL, seed0=EVAL_SEED0, scenario=SCENARIO)
    kills = [e["reward"] for e in eps]
    s = summarize(eps)
    lo, hi = bootstrap_ci(kills)
    text = (f"{name:<34} kills {s['reward_mean']:.1f} (95% CI {lo:.1f}..{hi:.1f}), tics {s['steps_mean']:.0f}, "
            f"agreement on unseen validation states {agreement * 100:.1f}%, "
            f"distinct results {distinct_outcomes(eps)}/{N_EVAL}")
    if baseline_kills is None:
        baseline_kills = kills
    else:
        d, dlo, dhi = paired_diff(kills, baseline_kills)
        text += f"; vs round 0 (paired): {d:+.2f} kills (CI {dlo:+.2f}..{dhi:+.2f})"
    say(text)

say("")
say(f"final model: {final_path} (hash {file_sha256(final_path)})")
say("notes: (1) the model of each round is chosen on the full validation set (best epoch), so the unseen-state "
    "figure is slightly optimistic; (2) seeds do not give independent games, so the real number of games is "
    "smaller than N_EVAL and the confidence intervals are optimistic; (3) a student trained with DAgger can "
    "match the teacher but not exceed it, because the teacher provides every label.")

os.makedirs("results", exist_ok=True)
with open(f"results/dagger_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"saved results/dagger_{SCENARIO}.txt", flush=True)
