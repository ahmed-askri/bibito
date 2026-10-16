import random

from bibito.scenarios import get_scenario

ACTIONS = list(get_scenario("basic").actions)  # still used by llm.py until it becomes scenario-aware


def make_random_policy(seed=0, scenario="basic"):
    rng = random.Random(seed)
    actions = list(get_scenario(scenario).actions)

    def policy(state):
        return rng.choice(actions)

    return policy


def always_attack(state):
    return "ATTACK"


def pick_target(enemies, group=1.15):
    """Among enemies at most `group` times farther than the nearest, the one that needs the least turning.

    `enemies` must be sorted by distance. Returns None when there are none.
    """
    if not enemies:
        return None
    nearest = enemies[0]["distance"]
    candidates = [e for e in enemies if e["distance"] <= nearest * group]
    return min(candidates, key=lambda e: abs(e["angle"]))


def make_scripted_policy(threshold=3.0, scenario="basic", group=1.15):
    """Aim at the target and shoot when it is within `threshold` degrees of straight ahead."""
    sc = get_scenario(scenario)

    def policy(state):
        target = pick_target(state["enemies"], group)
        if target is None:
            return sc.attack  # nothing to aim at; arbitrary default
        angle = target["angle"]
        if abs(angle) <= threshold:
            return sc.attack
        return sc.left if angle > 0 else sc.right

    return policy