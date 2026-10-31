# Bibito: a fast decision model for a DOOM agent, with an LLM as slow backup

A small, measured version of a "System One / System Two" idea: a tiny neural network (the fast brain)
learns from a slower decision-maker (the teacher) and answers in about 0.1 ms. An LLM is tested as the
slow backup. Built on ViZDoom (scenario `defend_the_center`).

Every number below was measured in this repository, with held-out test seeds and confidence intervals.

## Highlights

- **A 14-input MLP reaches about 97% of its teacher's score** on held-out test games
  (14.4 vs 14.9 kills) with a decision latency of 0.1-0.2 ms.
- **It is far faster and more reliable than an LLM in this task**: gpt-oss-20b took ~1.2 s per decision
  (budget: 114 ms) and agreed with the teacher on only 56% of unseen states, against 95% for the student.
- **Evaluation caught a data leak.** Different seeds often produce identical games, so a split by seed
  leaked: 66% of validation states were also in training. Checking exact-state overlap fixed it, and all
  agreement figures use unseen states only.
- **Full evaluation harness**: same-seed comparisons, bootstrap confidence intervals, paired differences,
  latency percentiles, MLflow tracking, unit tests and CI.

## Pipeline

1. **State -> JSON.** Game state becomes a small structured state: ammo, health, and visible enemies
   with distance and angle relative to the player.
2. **Teacher.** A scripted bot (aim at the best target, shoot within 5 degrees).
3. **Dataset.** (state, teacher action) pairs. Test seeds are never used for training or tuning.
4. **Student.** A small MLP (14 inputs -> 64 -> 64 -> 3 actions) trained to imitate the teacher.
5. **DAgger.** The student plays, the teacher labels the states the student visited, then retrain.
6. **Router.** The student's confidence decides when to escalate to an LLM.
7. **Evaluation harness.** Same seeds for every policy, confidence intervals, latency, tracking, CI.

## Results

`defend_the_center`, 100 games per policy, frame skip 4. Kills = monsters killed.
Held-out **test** seeds 9000-9099, each model evaluated once:

| Policy | Kills (95% CI) | vs teacher (paired) | Decision latency p50 |
|---|---|---|---|
| Scripted teacher | 14.9 (13.9-15.9) | - | ~0.002 ms |
| Student (MLP) | 14.4 (13.4-15.3) | -0.55 (-0.91 to -0.23) | ~0.1-0.2 ms |
| Student + DAgger (3 rounds) | 14.5 (13.4-15.5) | -0.44 (-0.97 to +0.03) | ~0.1-0.2 ms |

Agreement with the teacher on **unseen states** (offline):

| Model | Agreement |
|---|---|
| Student | 96.5% |
| Student + DAgger | 97.9% |
| LLM (gpt-oss-20b, 100 sampled rows) | 56.0% |

All policies die in every game: with 26 bullets and melee enemies that keep coming, surviving to the
time limit is not realistic, so the metric is kills.

## Key findings

- **The teacher is a hard baseline.** A 10-line rule is free (~0.002 ms) and strong. The LLM, given no
  rule, was slower and less accurate, so distillation into a small network was the better design here.
- **The student knows when it is unsure.** Its least-confident predictions are right only 36-59% of the
  time, against 96.5% overall.
- **An LLM fallback did not pay off in this scenario.** I escalated the k least-confident decisions to
  the LLM and measured exact accuracy on all 9,547 unseen validation rows. The LLM was weaker than the
  student on those rows in every bucket, so the router stayed at or below student-only accuracy
  (96.5% -> 96.0% at 3% escalation).
- **DAgger improved agreement but not score.** Agreement with the teacher on unseen states went from
  96.5% to 97.9% while kills stayed flat (test: 14.4 vs 14.5). Agreement is not game score, and a
  student trained on a teacher's labels is bounded by that teacher.

## Scope and next steps

- **Scope.** One game and two small scenarios (`basic` to build the pipeline, `defend_the_center` for
  the experiments), with a symbolic JSON state as input.
- **Statistics.** Different seeds often give identical games (about 50 distinct results per 100), so
  confidence intervals are indicative rather than exact.
- **Router evaluation.** Measured offline against the teacher's labels. Next: an in-game router test,
  a stronger LLM, and calibration (temperature scaling, expected calibration error).
- **Teacher.** Next: a PPO-trained teacher that could outperform the scripted rule, which is where
  distillation becomes most useful.
- **Pixels.** Next: a CNN student that plays from frames instead of the JSON state.
- **Deployment.** Next: FastAPI + Docker serving, a model registry, drift monitoring and a retrain loop.
- **LLM numbers** come from a free-tier provider with rate limits; token prices are taken from
  third-party trackers.

## Run it

Use Python 3.11 or 3.12 (ViZDoom has no wheel for 3.15).

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt mlflow
python -m pytest -q
python -u build_dataset.py defend_the_center 200
python -u train_student.py defend_the_center
python -u run_eval.py defend_the_center 100
python -u run_dagger.py
```

`data/`, `models/` and `results/` are not in the repository; the commands above regenerate them.
LLM experiments need a `GROQ_API_KEY` in `.env` (copy `.env.example`; never commit `.env`).

## Stack

Python, PyTorch, ViZDoom, MLflow, pytest, GitHub Actions. LLM calls through an OpenAI-compatible client.