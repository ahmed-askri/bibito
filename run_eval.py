import os
import sys

from bibito.evaluate import FRAME_SKIP, run_episodes, summarize
from bibito.harness import bootstrap_ci, distinct_outcomes, file_sha256, paired_diff
from bibito.policies import make_scripted_policy
from bibito.student import StudentPolicy, load_model

try:
    import mlflow
except ImportError:
    mlflow = None

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FINAL = "--final" in sys.argv
SCENARIO = args[0] if args else "defend_the_center"
N = int(args[1]) if len(args) > 1 else 100
SEED0 = 9000 if FINAL else 2000  # final = test seeds 9000-9099, touch only for the last numbers
if FINAL and N > 100:
    raise SystemExit("test seeds are 9000-9099: N must be <= 100")

MODEL_PATH = f"models/student_{SCENARIO}.pt"
model, scenario = load_model(MODEL_PATH)
players = {
    "teacher_t5": make_scripted_policy(5, SCENARIO),
    "student": StudentPolicy(model, scenario),
}
model_hash = file_sha256(MODEL_PATH)
tag = "TEST (final)" if FINAL else "validation"

lines = [f"scenario: {SCENARIO}, {tag} seeds {SEED0}..{SEED0 + N - 1}, {N} games, frame_skip {FRAME_SKIP}, "
         f"student file hash {model_hash}"]
print(lines[0], flush=True)

if mlflow:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"bibito-{SCENARIO}")
else:
    print("mlflow not installed: results go to the text file only (pip install mlflow)", flush=True)

rewards = {}
for name, policy in players.items():
    episodes = run_episodes(policy, n_episodes=N, seed0=SEED0, scenario=SCENARIO)
    s = summarize(episodes)
    r = [e["reward"] for e in episodes]
    rewards[name] = r
    lo, hi = bootstrap_ci(r)
    distinct = distinct_outcomes(episodes)
    line = (f"{name:<11} kills {s['reward_mean']:.1f} (95% CI {lo:.1f}..{hi:.1f}, std {s['reward_std']:.1f}), "
            f"died {s['death_rate'] * 100:.0f}%, tics {s['steps_mean']:.0f}, "
            f"latency p50 {s['latency_p50_ms']:.3f} ms p95 {s['latency_p95_ms']:.3f} ms, "
            f"distinct (kills,tics) results {distinct}/{N}")
    lines.append(line)
    print(line, flush=True)

    if mlflow:
        with mlflow.start_run(run_name=f"{name}-{'final' if FINAL else 'val'}"):
            mlflow.log_params({
                "scenario": SCENARIO, "policy": name, "n_games": N, "seed0": SEED0,
                "frame_skip": FRAME_SKIP, "final": FINAL,
                "student_hash": model_hash if name == "student" else "-",
            })
            mlflow.log_metrics({
                "kills_mean": s["reward_mean"], "kills_std": s["reward_std"],
                "kills_ci_lo": lo, "kills_ci_hi": hi,
                "death_rate": s["death_rate"], "tics_mean": s["steps_mean"],
                "latency_p50_ms": s["latency_p50_ms"], "latency_p95_ms": s["latency_p95_ms"],
                "distinct_outcomes": distinct,
            })

d, dlo, dhi = paired_diff(rewards["student"], rewards["teacher_t5"])
verdict = "the CI contains 0: no measurable difference" if dlo <= 0 <= dhi else "the CI excludes 0"
lines.append(f"student - teacher (same games, paired): {d:+.2f} kills (95% CI {dlo:+.2f}..{dhi:+.2f}); {verdict}")
lines.append("note: seeds do not give independent games here, so the real sample is smaller than N; "
             "treat the CI as optimistic.")
print("\n".join(lines[-2:]), flush=True)

os.makedirs("results", exist_ok=True)
out = f"results/eval_{SCENARIO}{'_final' if FINAL else ''}.txt"
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"saved {out}", flush=True)