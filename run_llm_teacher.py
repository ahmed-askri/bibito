import os
import sys

from bibito.evaluate import FRAME_SKIP, run_episodes, summarize
from bibito.llm import cost_usd, make_groq_policy
from bibito.policies import make_scripted_policy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED0 = 2000  # validation seeds
BUDGET_MS = FRAME_SKIP / 35 * 1000  # real-time budget per decision

out = []


def emit(line):
    print(line, flush=True)
    out.append(line)


def report(name, episodes):
    s = summarize(episodes)
    lat = [x for e in episodes for x in e["latencies_ms"]]
    over = sum(x > BUDGET_MS for x in lat) / len(lat)
    emit(f"{name}: reward {s['reward_mean']:.1f} +/- {s['reward_std']:.0f}, kill {s['kill_rate'] * 100:.0f}%, "
         f"tics {s['steps_mean']:.1f}, latency p50 {s['latency_p50_ms']:.1f} ms p95 {s['latency_p95_ms']:.1f} ms, "
         f"over budget: {over * 100:.0f}%")


emit(f"episodes={N}, seeds from {SEED0}, frame_skip={FRAME_SKIP}, real-time budget = {BUDGET_MS:.0f} ms per decision")
report("scripted_t5", run_episodes(make_scripted_policy(5), n_episodes=N, seed0=SEED0))

llm = make_groq_policy()
emit(f"model: {llm.model}")
report("llm", run_episodes(llm, n_episodes=N, seed0=SEED0,
                           log=lambda i, n, ep: print(f">> llm {i}/{n}", flush=True)))

st = llm.stats
decisions = max(1, st.calls + st.failures)
emit(f"calls={st.calls} retries={st.retries} invalid_replies={st.invalid} failures={st.failures}")
emit(f"retry reasons: {dict(st.retry_reasons)}")
emit(f"tokens per decision: in {st.tokens_in / decisions:.0f}, out {st.tokens_out / decisions:.0f}")

p_in, p_out = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
if p_in and p_out:
    cost = cost_usd(st, float(p_in), float(p_out))
    emit(f"estimated cost per 1000 decisions: ${cost / decisions * 1000:.4f} (prices: in {p_in}, out {p_out} USD per 1M tokens)")
else:
    emit("cost: set LLM_PRICE_IN and LLM_PRICE_OUT in .env (USD per 1M tokens, from the provider's pricing page)")

os.makedirs("results", exist_ok=True)
with open("results/teachers.txt", "w") as f:
    f.write("\n".join(out) + "\n")