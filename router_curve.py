import json
import os
import statistics
import sys
import time

import torch
from dotenv import load_dotenv

from bibito.dataset import load_jsonl
from bibito.llm import make_groq_policy
from bibito.router import escalation_curve, least_confident
from bibito.scenarios import get_scenario
from bibito.student import load_model, rows_to_xy

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CACHE = f"data/llm_cache_{SCENARIO}.jsonl"
load_dotenv()


def key(row):
    return json.dumps(row["state"], sort_keys=True)


train_keys = {key(r) for r in load_jsonl(f"data/{SCENARIO}_train.jsonl")}
val = [r for r in load_jsonl(f"data/{SCENARIO}_val.jsonl") if key(r) not in train_keys]  # unseen states only

model, scenario = load_model(f"models/student_{SCENARIO}.pt")
actions = list(get_scenario(scenario).actions)
x, y = rows_to_xy(val, actions)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)  # raw confidence: calibration was skipped
conf_t, pred = probs.max(dim=1)
conf = conf_t.tolist()
student_correct = (pred == y).tolist()
lowest = least_confident(conf, K_MAX)

cache = {}
if os.path.exists(CACHE):
    for rec in load_jsonl(CACHE):
        cache[rec["key"]] = rec

todo = []
for i in lowest:
    k_ = key(val[i])
    if k_ not in cache and k_ not in todo:
        todo.append(k_)
by_key = {key(val[i]): val[i] for i in lowest}
print(f"{len(val)} unseen rows; {K_MAX} least confident = {len({key(val[i]) for i in lowest})} distinct states; "
      f"{len(todo)} LLM calls needed ({len(cache)} cached)", flush=True)

if todo:
    llm = make_groq_policy(scenario=SCENARIO)
    with open(CACHE, "a", encoding="utf-8") as f:
        for n, k_ in enumerate(todo, 1):
            before = (llm.stats.tokens_in, llm.stats.tokens_out)
            t0 = time.perf_counter()
            action = llm(by_key[k_]["state"])
            rec = {"key": k_, "action": action, "ms": (time.perf_counter() - t0) * 1000,
                   "tin": llm.stats.tokens_in - before[0], "tout": llm.stats.tokens_out - before[1]}
            cache[k_] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f">> {n}/{len(todo)}", flush=True)

llm_correct = {i: cache[key(val[i])]["action"] == val[i]["action"] for i in lowest}
ks = [k for k in (0, 25, 50, 100, 200, 300, 500, 1000) if k <= K_MAX]
curve = escalation_curve(conf, student_correct, llm_correct, ks)

recs = [cache[key(val[i])] for i in lowest]
p50 = statistics.median(r["ms"] for r in recs)
p_in, p_out = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
cost_call = None
if p_in and p_out:
    cost_call = sum(r["tin"] / 1e6 * float(p_in) + r["tout"] / 1e6 * float(p_out) for r in recs) / len(recs)

lines = [
    f"scenario: {SCENARIO}; {len(val)} unseen validation rows; student-only accuracy {sum(student_correct) / len(val) * 100:.2f}%",
    f"confidence = raw softmax (no calibration); LLM latency p50 {p50:.0f} ms per escalated call (includes rate-limit waits)",
    "",
    "  k escalated   share   conf<=   router acc   student on those   LLM on those   extra latency/decision   cost per 1000 decisions",
]
for row in curve:
    thr = f"{row['threshold']:.3f}" if row["threshold"] is not None else "  -  "
    s_acc = f"{row['student_acc_on_escalated'] * 100:5.1f}%" if row["k"] else "  -  "
    l_acc = f"{row['llm_acc_on_escalated'] * 100:5.1f}%" if row["k"] else "  -  "
    cost = f"${row['escalated'] * 1000 * cost_call:.4f}" if cost_call is not None else "n/a"
    lines.append(f"  {row['k']:11d}  {row['escalated'] * 100:5.2f}%  {thr}   {row['router_acc'] * 100:8.2f}%   "
                 f"{s_acc:>16}   {l_acc:>12}   {row['escalated'] * p50:14.1f} ms   {cost:>12}")

print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/router_curve_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")