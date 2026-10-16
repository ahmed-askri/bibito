import json
import os
import random
import sys
import time

import torch

from bibito.dataset import load_jsonl
from bibito.evaluate import percentile
from bibito.llm import cost_usd, make_groq_policy
from bibito.student import StudentPolicy, load_model

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def key(row):
    return json.dumps(row["state"], sort_keys=True)


def contested(row):
    es = row["state"]["enemies"]
    return sum(e["distance"] <= es[0]["distance"] * 1.15 for e in es) > 1


train_keys = {key(r) for r in load_jsonl(f"data/{SCENARIO}_train.jsonl")}
val = [r for r in load_jsonl(f"data/{SCENARIO}_val.jsonl") if key(r) not in train_keys]  # unseen states only
rng = random.Random(0)
hard = [r for r in val if contested(r)]
easy = [r for r in val if not contested(r)]
rows = rng.sample(hard, min(N // 2, len(hard))) + rng.sample(easy, min(N - N // 2, len(easy)))

model, scenario = load_model(f"models/student_{SCENARIO}.pt")
student = StudentPolicy(model, scenario)
llm = make_groq_policy(scenario=SCENARIO)
print(f"model: {llm.model}, {len(rows)} unseen validation rows ({len(hard)} contested available)", flush=True)

results, latencies = [], []
for i, r in enumerate(rows):
    t0 = time.perf_counter()
    a_llm = llm(r["state"])
    latencies.append((time.perf_counter() - t0) * 1000)
    results.append({"truth": r["action"], "llm": a_llm, "student": student(r["state"]), "hard": contested(r)})
    if (i + 1) % 10 == 0:
        print(f">> {i + 1}/{len(rows)}", flush=True)


def rate(items, who):
    return sum(x[who] == x["truth"] for x in items) / len(items) * 100 if items else float("nan")


hard_items = [x for x in results if x["hard"]]
easy_items = [x for x in results if not x["hard"]]
student_wrong = [x for x in results if x["student"] != x["truth"]]
llm_wrong = [x for x in results if x["llm"] != x["truth"]]

lines = [
    f"scenario: {SCENARIO}, model: {llm.model}, {len(results)} unseen validation rows (half contested)",
    f"agreement with teacher   student {rate(results, 'student'):.1f}%   llm {rate(results, 'llm'):.1f}%",
    f"  contested rows ({len(hard_items)})   student {rate(hard_items, 'student'):.1f}%   llm {rate(hard_items, 'llm'):.1f}%",
    f"  other rows ({len(easy_items)})       student {rate(easy_items, 'student'):.1f}%   llm {rate(easy_items, 'llm'):.1f}%",
    f"rows where the student is wrong: {len(student_wrong)}; the LLM is right on {rate(student_wrong, 'llm'):.1f}% of them",
    f"rows where the LLM is wrong: {len(llm_wrong)}; the student is right on {rate(llm_wrong, 'student'):.1f}% of them",
    f"llm latency per decision (includes retry waits): p50 {percentile(latencies, 50):.0f} ms, p95 {percentile(latencies, 95):.0f} ms (budget 114 ms)",
]
st = llm.stats
decisions = max(1, st.calls + st.failures)
lines.append(f"calls={st.calls} retries={st.retries} invalid_replies={st.invalid} failures={st.failures} retry reasons: {dict(st.retry_reasons)}")
lines.append(f"tokens per decision: in {st.tokens_in / decisions:.0f}, out {st.tokens_out / decisions:.0f}")
p_in, p_out = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
if p_in and p_out:
    lines.append(f"estimated cost per 1000 decisions: ${cost_usd(st, float(p_in), float(p_out)) / decisions * 1000:.4f}")

print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/llm_agreement_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")