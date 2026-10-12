import time

from bibito.llm import make_groq_policy

llm = make_groq_policy()
print("model:", llm.model)
state = {"ammo": 50, "enemies": [{"name": "Cacodemon", "distance": 385.0, "angle": 12.0}]}
t0 = time.perf_counter()
action = llm(state)
print(f"action={action} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
print("raw reply:", repr(llm.last_reply))
print(llm.stats)