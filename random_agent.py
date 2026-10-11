import os
import random
import time

import vizdoom as vzd

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.set_window_visible(True)
game.init()

buttons = game.get_available_buttons()
print("buttons:", [b.name for b in buttons])
print("game variables:", [v.name for v in game.get_available_game_variables()])

# one action = one button pressed at a time
n = len(buttons)
actions = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

for ep in range(5):
    game.new_episode()
    steps = 0
    while not game.is_episode_finished():
        game.make_action(random.choice(actions))
        steps += 1
        time.sleep(0.02)  # slow down so you can watch
    print(f"episode {ep}: steps={steps} total_reward={game.get_total_reward()}")

game.close()