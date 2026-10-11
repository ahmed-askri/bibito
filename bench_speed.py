import os
import time

import vizdoom as vzd


def make(variant):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    if variant == "small_res":
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    if variant == "no_extras":
        game.set_render_hud(False)
        game.set_render_weapon(False)
        game.set_render_crosshair(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
    if variant == "vsync0":
        game.add_game_args("+vid_vsync 0")
    game.init()
    return game


def measure(game, skip, episodes=2):
    tics = 0
    t0 = time.perf_counter()
    for _ in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            game.make_action([0, 0, 1], skip)  # attack, repeated for `skip` tics
        tics += game.get_episode_time()
    return tics / (time.perf_counter() - t0)


for variant, skip in [("default", 1), ("skip4", 4), ("small_res", 1), ("no_extras", 1), ("vsync0", 1)]:
    game = make(variant)
    print(f"{variant:<10} skip={skip}: {measure(game, skip):7.0f} tics/s", flush=True)
    game.close()