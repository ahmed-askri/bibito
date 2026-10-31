import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bibito.evaluate import make_game
from bibito.scenarios import get_scenario
from bibito.state import make_actions, state_from_game
from bibito.student import StudentPolicy, load_model

SCENARIO = "defend_the_center"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SECONDS = 10      # durée du GIF
FPS = 17          # une image toutes les 2 tics de jeu (le jeu tourne à 35 tics par seconde)
SCALE = 1.5

sc = get_scenario(SCENARIO)
model, name = load_model(f"models/student_{SCENARIO}.pt")
policy = StudentPolicy(model, name)
game = make_game(sc)
names, vectors = make_actions([b.name for b in game.get_available_buttons()])
index = {n: i for i, n in enumerate(names)}
font = ImageFont.load_default(size=15)


def to_image(buf):
    a = np.asarray(buf)
    if a.ndim == 3 and a.shape[0] == 3:  # canaux en premier -> canaux en dernier
        a = np.transpose(a, (1, 2, 0))
    return Image.fromarray(a.astype("uint8")).convert("RGB")


def decorate(img, action, kills):
    img = img.resize((int(img.width * SCALE), int(img.height * SCALE)), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 28], fill=(20, 20, 19))
    d.text((8, 5), f"Bibito student   action: {action}   kills: {int(kills)}", font=font, fill=(240, 239, 236))
    return img


game.set_seed(SEED)
game.new_episode()
frames = []
max_frames = SECONDS * FPS
while not game.is_episode_finished() and len(frames) < max_frames:
    action = policy(state_from_game(game, game.get_state(), sc.enemies))
    vec = vectors[index[action]]
    for _ in range(2):  # une décision = 4 tics = 2 images, comme dans l'évaluation
        if game.is_episode_finished():
            break
        frames.append(decorate(to_image(game.get_state().screen_buffer), action, game.get_total_reward()))
        game.make_action(vec, 2)
game.close()

os.makedirs("docs", exist_ok=True)
small = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
small[0].save("docs/demo.gif", save_all=True, append_images=small[1:], duration=int(1000 / FPS), loop=0, optimize=True)
print(f"saved docs/demo.gif: {len(frames)} frames, seed {SEED}, size {os.path.getsize('docs/demo.gif') / 1e6:.1f} MB", flush=True)