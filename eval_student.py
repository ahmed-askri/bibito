import os
import sys

from bibito.evaluate import run_episodes, summarize
from bibito.policies import make_scripted_policy
from bibito.student import StudentPolicy, load_model

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEED0 = 2000  # validation seeds, same for both players

model, scenario = load_model(f"models/student_{SCENARIO}.pt")
players = {
    "scripted_t5 (teacher)": make_scripted_policy(5, SCENARIO),
    "student": StudentPolicy(model, scenario),
}

lines = [f"scenario: {SCENARIO}, {N} games, seeds from {SEED0}"]
print(lines[0], flush=True)
for name, policy in players.items():
    s = summarize(run_episodes(policy, n_episodes=N, seed0=SEED0, scenario=SCENARIO))
    lines.append(f"{name:<22} reward {s['reward_mean']:.1f} +/- {s['reward_std']:.1f}, died {s['death_rate'] * 100:.0f}%, "
                 f"tics {s['steps_mean']:.0f}, latency p50 {s['latency_p50_ms']:.3f} ms p95 {s['latency_p95_ms']:.3f} ms")
    print(lines[-1], flush=True)

os.makedirs("results", exist_ok=True)
with open(f"results/student_eval_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")