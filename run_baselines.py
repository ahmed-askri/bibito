import os
import time

from bibito.evaluate import FRAME_SKIP, run_episodes, summarize
from bibito.policies import always_attack, make_random_policy, make_scripted_policy

contenders = {
    "random": make_random_policy(0),
    "always_attack": always_attack,
    "scripted_t1": make_scripted_policy(1),
    "scripted_t2": make_scripted_policy(2),
    "scripted_t3": make_scripted_policy(3),
    "scripted_t5": make_scripted_policy(5),
    "scripted_t8": make_scripted_policy(8),
    "scripted_t12": make_scripted_policy(12),
}


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
header = f"{'policy':<14}{'reward':>14}{'kill%':>8}{'steps':>8}{'p50ms':>8}{'p95ms':>8}"
lines = [f"frame_skip={FRAME_SKIP}", header]
print(header, flush=True)

for name, policy in contenders.items():
    s = summarize(run_episodes(policy, n_episodes=50, log=make_progress(name)))
    line = (f"{name:<14}{s['reward_mean']:>8.1f}+/-{s['reward_std']:<4.0f}"
            f"{s['kill_rate'] * 100:>7.0f}%{s['steps_mean']:>8.1f}"
            f"{s['latency_p50_ms']:>8.3f}{s['latency_p95_ms']:>8.3f}")
    lines.append(line)
    print(line, flush=True)
    with open("results/baselines.txt", "w") as f:
        f.write("\n".join(lines) + "\n")