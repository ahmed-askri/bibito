import os

import torch

from bibito.dataset import load_jsonl
from bibito.policies import ACTIONS
from bibito.student import rows_to_xy, save_model, train

train_rows = load_jsonl("data/train.jsonl")
val_rows = load_jsonl("data/val.jsonl")


def log(epoch, loss, val_acc):
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.3f}  val_acc {val_acc:.3f}", flush=True)


model, best = train(train_rows, val_rows, log=log)
save_model(model, "models/student.pt")

xv, yv = rows_to_xy(val_rows)
with torch.no_grad():
    pred = model(xv).argmax(dim=1)

lines = [f"best val agreement with teacher: {best * 100:.1f}% ({len(val_rows)} validation rows)"]
for i, name in enumerate(ACTIONS):
    mask = yv == i
    acc = (pred[mask] == yv[mask]).float().mean().item() if mask.any() else float("nan")
    lines.append(f"  {name}: {acc * 100:.1f}% of {int(mask.sum())} rows")

near = torch.tensor([4 <= abs(r["state"]["enemies"][0]["angle"]) <= 6 for r in val_rows])
near_acc = (pred[near] == yv[near]).float().mean().item()
lines.append(f"near the threshold (4 <= |angle| <= 6): {near_acc * 100:.1f}% of {int(near.sum())} rows")

print("\n".join(lines), flush=True)
os.makedirs("results", exist_ok=True)
with open("results/student.txt", "w") as f:
    f.write("\n".join(lines) + "\n")