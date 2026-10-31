import json
import random


def state_key(state):
    """Exact identity of a state, so two identical states can be matched across datasets."""
    return json.dumps(state, sort_keys=True)


def unseen_rows(rows, reference_rows):
    """Rows whose exact state does not appear in reference_rows (the honest validation set)."""
    seen = {state_key(r["state"]) for r in reference_rows}
    return [r for r in rows if state_key(r["state"]) not in seen]


class DAggerRollout:
    """Plays with the student (plus some random noise) and labels every state it visits with the teacher.

    This is the DAgger step: the states come from the student's own play, the labels come from the teacher.
    States with no enemies are not recorded (the teacher's answer there is an arbitrary default),
    the same rule as dataset.Recorder.
    """

    def __init__(self, student, teacher, actions, seed0, round_id, noise=0.1, rng_seed=0):
        self.student = student
        self.teacher = teacher
        self.actions = list(actions)
        self.seed0 = seed0
        self.round_id = round_id
        self.noise = noise
        self.rng = random.Random(rng_seed)
        self.episode = 0
        self.step = 0
        self.rows = []

    def __call__(self, state):
        label = self.teacher(state)
        if state["enemies"]:
            self.rows.append({
                "seed": self.seed0 + self.episode,
                "step": self.step,
                "round": self.round_id,
                "state": state,
                "action": label,
            })
        self.step += 1
        if self.rng.random() < self.noise:
            return self.rng.choice(self.actions)  # noise: visit states the student alone would not reach
        return self.student(state)

    def end_episode(self):
        self.episode += 1
        self.step = 0
