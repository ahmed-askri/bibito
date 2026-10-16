import json
import os
import random
import sys
from collections import Counter

import vizdoom as vzd

from bibito.scenarios import get_scenario
from bibito.state import make_actions, state_from_game

sc = get_scenario(sys.argv[1] if len(sys.argv) > 1 else "defend_the_center")

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, sc.cfg))
game.set_window_visible(False)
game.set_objects_info_enabled(True)
have = list(game.get_available_game_variables())
for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
          vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH):
    if v not in have:
        game.add_available_game_variable(v)
game.init()

names, vectors = make_actions([b.name for b in game.get_available_buttons()])
assert tuple(names) == sc.actions, f"game buttons {names} differ from the scenario config {sc.actions}"
print("actions:", names)

game.set_seed(2000)
game.new_episode()
lines = []
while not game.is_episode_finished():
    lines.append(json.dumps(state_from_game(game, game.get_state(), sc.enemies)))
    game.make_action(random.choice(vectors), 4)
game.close()

print(len(lines), "decisions")
print("enemies per state:", dict(Counter(len(json.loads(x)["enemies"]) for x in lines)))
print("\n".join(lines[:3]))
print("...")
print("\n".join(lines[-2:]))