import os

from bibito.evaluate import run_episodes, summarize
from bibito.policies import make_scripted_policy
from bibito.student import StudentPolicy, load_model

N, SEED0 = 50, 2000  # validation seeds, same for both players

players = {
    "scripted_t5 (teacher)": make_scripted_policy(5),
    "student": StudentPolicy(load_model("models/student.pt")),
}

lines = []
for name, policy in players.items():
    s = summarize(run_episodes(policy, n_episodes=N, seed0=SEED0))
    lines.append(f"{name:<22} reward {s['reward_mean']:.1f} +/- {s['reward_std']:.0f}, kill {s['kill_rate'] * 100:.0f}%, "
                 f"tics {s['steps_mean']:.1f}, latency p50 {s['latency_p50_ms']:.3f} ms p95 {s['latency_p95_ms']:.3f} ms")
    print(lines[-1], flush=True)

os.makedirs("results", exist_ok=True)
with open("results/student_eval.txt", "w") as f:
    f.write("\n".join(lines) + "\n")