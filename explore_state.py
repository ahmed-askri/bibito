import os

import vizdoom as vzd

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.set_window_visible(True)

# extra info, must be set BEFORE init()
game.set_labels_buffer_enabled(True)   # which objects are visible on screen
game.set_objects_info_enabled(True)    # list of objects with positions
for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
    game.add_available_game_variable(v)

game.init()

buttons = game.get_available_buttons()
n = len(buttons)
actions = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
print("buttons:", [b.name for b in buttons])
print("variables:", [v.name for v in game.get_available_game_variables()])
print("episode timeout (tics):", game.get_episode_timeout())

game.new_episode()
for step in range(15):
    state = game.get_state()
    if state is None:
        break
    print(f"\n--- step {step}, tic {state.tic} ---")
    print("game_variables:", state.game_variables)
    print("screen shape:", state.screen_buffer.shape)
    for lab in state.labels:
        print("label:", lab.object_name, lab.object_position_x, lab.object_position_y, lab.object_angle)
    for obj in state.objects:
        print("object:", obj.name, obj.position_x, obj.position_y)
    reward = game.make_action(actions[step % n])
    print("action:", actions[step % n], "reward:", reward)

game.close()