import math

ENEMY_NAMES = {"Cacodemon"}  # extend per scenario (we'll check names later)


def normalize_angle(deg):
    """Map any angle to [-180, 180)."""
    return (deg + 180) % 360 - 180


def build_state(ammo, px, py, pangle, objects, enemy_names=ENEMY_NAMES):
    """objects: list of (name, x, y). Returns the JSON-ready state."""
    enemies = []
    for name, ox, oy in objects:
        if name not in enemy_names:
            continue
        dx, dy = ox - px, oy - py
        distance = math.hypot(dx, dy)
        angle = normalize_angle(math.degrees(math.atan2(dy, dx)) - pangle)
        enemies.append({"name": name, "distance": round(distance, 1), "angle": round(angle, 1)})
    enemies.sort(key=lambda e: e["distance"])
    return {"ammo": ammo, "enemies": enemies}


def state_from_game(game, state, enemy_names=ENEMY_NAMES):
    names = [v.name for v in game.get_available_game_variables()]
    v = dict(zip(names, state.game_variables))
    objects = [(o.name, o.position_x, o.position_y) for o in state.objects]
    return build_state(v["AMMO2"], v["POSITION_X"], v["POSITION_Y"], v["ANGLE"], objects, enemy_names)


def make_actions(button_names):
    """One action = press exactly one button. Returns (names, vectors)."""
    n = len(button_names)
    vectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return list(button_names), vectors