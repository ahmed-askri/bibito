import os
import sys
import time

from bibito.evaluate import FRAME_SKIP, run_episodes, summarize
from bibito.policies import always_attack, make_random_policy, make_scripted_policy

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "basic"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SEED0 = 1000 if SCENARIO == "basic" else 2000  # defend_the_center starts clean on the validation seeds
THRESHOLDS = {"basic": [1, 2, 3, 5, 8, 12], "defend_the_center": [3, 5, 8, 12, 20]}[SCENARIO]

contenders = {"random": make_random_policy(0, SCENARIO), "always_attack": always_attack}
for t in THRESHOLDS:
    contenders[f"scripted_t{t}"] = make_scripted_policy(t, SCENARIO)


def make_progress(name):
    t0 = time.perf_counter()
    total_steps = [0]

    def log(i, n, ep):
        total_steps[0] += ep["steps"]
        if i == 1 or i % 10 == 0 or i == n:
            dt = time.perf_counter() - t0
            print(f">> {name}: {i}/{n} episodes, {total_steps[0] / dt:.0f} tics/s", flush=True)

    return log


os.makedirs("results", exist_ok=True)
header = f"{'policy':<16}{'reward':>14}{'died%':>8}{'tics':>8}{'p50ms':>8}{'p95ms':>8}"
lines = [f"scenario={SCENARIO} episodes={N} seeds from {SEED0} frame_skip={FRAME_SKIP}", header]
print(lines[0], flush=True)
print(header, flush=True)

for name, policy in contenders.items():
    s = summarize(run_episodes(policy, n_episodes=N, seed0=SEED0, scenario=SCENARIO, log=make_progress(name)))
    line = (f"{name:<16}{s['reward_mean']:>8.1f}+/-{s['reward_std']:<4.1f}"
            f"{s['death_rate'] * 100:>7.0f}%{s['steps_mean']:>8.0f}"
            f"{s['latency_p50_ms']:>8.3f}{s['latency_p95_ms']:>8.3f}")
    lines.append(line)
    print(line, flush=True)
    with open(f"results/baselines_{SCENARIO}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")