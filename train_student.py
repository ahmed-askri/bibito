import os
import sys

import torch

from bibito.dataset import load_jsonl
from bibito.policies import pick_target
from bibito.scenarios import get_scenario
from bibito.student import rows_to_xy, save_model, train

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
actions = list(get_scenario(SCENARIO).actions)

train_rows = load_jsonl(f"data/{SCENARIO}_train.jsonl")
val_rows = load_jsonl(f"data/{SCENARIO}_val.jsonl")


def log(epoch, loss, val_acc):
    if epoch % 5 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.3f}  val_acc {val_acc:.3f}", flush=True)


model, best = train(train_rows, val_rows, actions, log=log)
save_model(model, f"models/student_{SCENARIO}.pt", SCENARIO)

xv, yv = rows_to_xy(val_rows, actions)
with torch.no_grad():
    pred = model(xv).argmax(dim=1)


def part(name, mask):
    n = int(mask.sum())
    acc = (pred[mask] == yv[mask]).float().mean().item() if n else float("nan")
    return f"  {name}: {acc * 100:.1f}% of {n} rows"


lines = [f"scenario: {SCENARIO}", f"best val agreement with teacher: {best * 100:.1f}% ({len(val_rows)} validation rows)"]
for i, name in enumerate(actions):
    lines.append(part(name, yv == i))

near = torch.tensor([abs(abs(pick_target(r["state"]["enemies"])["angle"]) - 5) <= 2 for r in val_rows])
lines.append(part("near the threshold", near))
cont = torch.tensor([sum(e["distance"] <= r["state"]["enemies"][0]["distance"] * 1.15
                         for e in r["state"]["enemies"]) > 1 for r in val_rows])
lines.append(part("contested (several enemies at similar distance)", cont))

print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/student_{SCENARIO}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")