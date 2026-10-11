from bibito.evaluate import run_episodes, summarize
from bibito.policies import always_attack, make_random_policy, make_scripted_policy

contenders = {
    "random": make_random_policy(0),
    "always_attack": always_attack,
    "scripted_t1": make_scripted_policy(1),
    "scripted_t2": make_scripted_policy(2),
    "scripted_t3": make_scripted_policy(3),
    "scripted_t5": make_scripted_policy(5),
}

print(f"{'policy':<14}{'reward':>14}{'kill%':>8}{'steps':>8}{'p50ms':>8}{'p95ms':>8}")
for name, policy in contenders.items():
    s = summarize(run_episodes(policy, n_episodes=50))
    print(f"{name:<14}{s['reward_mean']:>8.1f}+/-{s['reward_std']:<4.0f}"
          f"{s['kill_rate'] * 100:>7.0f}%{s['steps_mean']:>8.1f}"
          f"{s['latency_p50_ms']:>8.3f}{s['latency_p95_ms']:>8.3f}")