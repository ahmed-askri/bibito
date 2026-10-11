import math
import os
import statistics
import time

import vizdoom as vzd

from bibito.state import make_actions, state_from_game

FRAME_SKIP = 4  # the agent decides once every FRAME_SKIP game tics


def percentile(values, q):
    """Nearest-rank percentile."""
    ordered = sorted(values)
    k = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[k]


def summarize(episodes):
    rewards = [e["reward"] for e in episodes]
    lat = [x for e in episodes for x in e["latencies_ms"]]
    return {
        "episodes": len(episodes),
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "kill_rate": sum(e["killed"] for e in episodes) / len(episodes),
        "steps_mean": statistics.mean(e["steps"] for e in episodes),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }


def make_game(scenario="basic.cfg", visible=False):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario))
    game.set_window_visible(visible)
    game.set_objects_info_enabled(True)
    for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
        game.add_available_game_variable(v)
    game.init()
    return game


def run_episodes(policy, n_episodes=50, seed0=1000, scenario="basic.cfg", visible=False,
                 frame_skip=FRAME_SKIP, log=None):
    """log(i_done, n_total, episode_dict) is called after each episode, if given."""
    game = make_game(scenario, visible)
    names, vectors = make_actions([b.name for b in game.get_available_buttons()])
    index = {n: i for i, n in enumerate(names)}
    timeout = game.get_episode_timeout()
    assert timeout > 0, "this scenario has no timeout; 'killed' detection needs one"

    episodes = []
    for i in range(n_episodes):
        game.set_seed(seed0 + i)
        game.new_episode()
        latencies = []
        while not game.is_episode_finished():
            s = state_from_game(game, game.get_state())
            t0 = time.perf_counter()
            action = policy(s)
            latencies.append((time.perf_counter() - t0) * 1000)
            game.make_action(vectors[index[action]], frame_skip)
        tics = len(latencies) * frame_skip  # get_episode_time() reads 0 after a kill, so we count ourselves
        episodes.append({
            "reward": game.get_total_reward(),
            "steps": tics,  # game tics
            "killed": tics < timeout,  # episode ended before the timeout = monster died
            "latencies_ms": latencies,
        })
        if log:
            log(i + 1, n_episodes, episodes[-1])
    game.close()
    return episodes