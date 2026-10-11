import random

ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]


def make_random_policy(seed=0):
    rng = random.Random(seed)

    def policy(state):
        return rng.choice(ACTIONS)

    return policy


def always_attack(state):
    return "ATTACK"


def make_scripted_policy(threshold=3.0):
    """Shoot when the nearest enemy is within `threshold` degrees, else strafe toward it."""

    def policy(state):
        if not state["enemies"]:
            return "ATTACK"  # nothing to aim at; arbitrary default
        angle = state["enemies"][0]["angle"]
        if abs(angle) <= threshold:
            return "ATTACK"
        return "MOVE_LEFT" if angle > 0 else "MOVE_RIGHT"

    return policy