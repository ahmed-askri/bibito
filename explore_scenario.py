import os
import random
import sys
from collections import Counter

import vizdoom as vzd

scenario = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center.cfg"
print("available scenarios:", sorted(f for f in os.listdir(vzd.scenarios_path) if f.endswith(".cfg")))

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, scenario))
game.set_window_visible(False)
game.set_objects_info_enabled(True)
have = list(game.get_available_game_variables())
for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
          vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH):
    if v not in have:
        game.add_available_game_variable(v)
game.init()

buttons = [b.name for b in game.get_available_buttons()]
var_names = [v.name for v in game.get_available_game_variables()]
print("buttons:", buttons)
print("variables:", var_names)
print("episode timeout:", game.get_episode_timeout())

n = len(buttons)
actions = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
seen = Counter()

for ep in range(3):
    game.set_seed(2000 + ep)
    game.new_episode()
    decisions = 0
    while not game.is_episode_finished():
        s = game.get_state()
        v = dict(zip(var_names, s.game_variables))
        for o in s.objects:
            seen[o.name] += 1
        show = ep == 0 and decisions < 12
        if show:
            objs = [(o.name, round(o.position_x), round(o.position_y)) for o in s.objects]
            print(f"step {decisions}: {v}")
            print(f"   {len(objs)} objects, first ones: {objs[:8]}")
        reward = game.make_action(random.choice(actions), 4)
        if show:
            print(f"   reward {reward}")
        decisions += 1
    print(f"episode {ep}: {decisions} decisions ({decisions * 4} tics), "
          f"total reward {game.get_total_reward()}, player dead: {game.is_player_dead()}")

print("object names seen (count):", dict(seen))
game.close()