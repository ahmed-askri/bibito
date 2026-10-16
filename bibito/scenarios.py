from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    cfg: str            # ViZDoom config file
    enemies: frozenset  # object names that count as enemies
    actions: tuple      # button names, in the order the game lists them
    left: str           # action that brings an enemy at a positive angle toward the center
    right: str          # same for a negative angle
    attack: str = "ATTACK"


SCENARIOS = {
    "basic": Scenario(
        "basic", "basic.cfg", frozenset({"Cacodemon"}),
        ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"), "MOVE_LEFT", "MOVE_RIGHT",
    ),
    "defend_the_center": Scenario(
        "defend_the_center", "defend_the_center.cfg", frozenset({"Demon", "MarineChainsawVzd"}),
        ("TURN_LEFT", "TURN_RIGHT", "ATTACK"), "TURN_LEFT", "TURN_RIGHT",
    ),
}


def get_scenario(name):
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]