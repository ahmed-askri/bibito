import json
import os
import random

import vizdoom as vzd

from bibito.state import make_actions, state_from_game

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.set_window_visible(False)
game.set_objects_info_enabled(True)
for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
    game.add_available_game_variable(v)
game.init()

names, vectors = make_actions([b.name for b in game.get_available_buttons()])
print("actions:", names)

game.new_episode()
lines = []
while not game.is_episode_finished():
    lines.append(json.dumps(state_from_game(game, game.get_state())))
    game.make_action(random.choice(vectors))
game.close()

print("\n".join(lines[:5]))
print("...")
print("\n".join(lines[-3:]))