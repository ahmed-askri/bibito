import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from bibito.policies import ACTIONS

SYSTEM_PROMPT = (
    "You control a player in a DOOM room. Each turn you receive the game state as JSON "
    "and must choose exactly ONE action.\n"
    "Actions: MOVE_LEFT (strafe left), MOVE_RIGHT (strafe right), ATTACK (fire the pistol).\n"
    "Each enemy has 'distance' and 'angle' (degrees) relative to where you face: "
    "positive angle = enemy is to your left, negative = to your right, 0 = straight ahead.\n"
    "Shots only hit an enemy that is almost straight ahead. You cannot turn, only strafe. "
    "Every turn costs points, so kill the enemy as fast as you can.\n"
    "Reply with exactly one word: MOVE_LEFT, MOVE_RIGHT or ATTACK."
)


@dataclass
class LLMStats:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    invalid: int = 0   # replies we could not parse
    failures: int = 0  # calls that failed even after retries
    retry_reasons: Counter = field(default_factory=Counter)  # error type -> count


def parse_action(text):
    """Return the one action named in `text`, or None if zero or several are named."""
    t = (text or "").upper().replace("MOVE LEFT", "MOVE_LEFT").replace("MOVE RIGHT", "MOVE_RIGHT")
    found = [a for a in ACTIONS if re.search(rf"\b{a}\b", t)]
    return found[0] if len(found) == 1 else None


def cost_usd(stats, price_in_per_m, price_out_per_m):
    return stats.tokens_in / 1e6 * price_in_per_m + stats.tokens_out / 1e6 * price_out_per_m


class LLMPolicy:
    def __init__(self, client, model, retry_on=(), max_retries=5, sleep=time.sleep,
                 fallback="ATTACK", extra_args=None, max_completion_tokens=300):
        self.client = client
        self.model = model
        self.retry_on = retry_on
        self.max_retries = max_retries
        self.sleep = sleep
        self.fallback = fallback
        self.extra_args = extra_args or {}
        self.max_completion_tokens = max_completion_tokens
        self.stats = LLMStats()
        self.last_reply = None

    def __call__(self, state):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(state)},
        ]
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                    **self.extra_args,
                )
                break
            except self.retry_on as err:
                self.stats.retry_reasons[type(err).__name__] += 1
                if attempt == self.max_retries:
                    self.stats.failures += 1
                    return self.fallback
                self.stats.retries += 1
                self.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s, ...

        self.stats.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.tokens_in += usage.prompt_tokens
            self.stats.tokens_out += usage.completion_tokens
        self.last_reply = resp.choices[0].message.content or ""
        action = parse_action(self.last_reply)
        if action is None:
            self.stats.invalid += 1
            return self.fallback
        return action


def make_groq_policy(model=None):
    from dotenv import load_dotenv
    import openai

    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing: put it in .env")
    client = openai.OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30,
        max_retries=0,  # we do our own retries so we can count them
    )
    model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}  # [verify] accepted by the provider
    retry_on = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
    return LLMPolicy(client, model, retry_on=retry_on, extra_args=extra)