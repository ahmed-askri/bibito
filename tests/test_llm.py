import json
from types import SimpleNamespace

import pytest

from bibito.llm import LLMPolicy, LLMStats, cost_usd, parse_action


class Boom(Exception):
    pass


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
        )


STATE = {"ammo": 50, "enemies": [{"name": "Cacodemon", "distance": 385.0, "angle": 12.0}]}


def make(script, **kw):
    sleeps = []
    client = FakeClient(script)
    policy = LLMPolicy(client, "fake-model", retry_on=(Boom,), sleep=sleeps.append, **kw)
    return policy, client, sleeps


def test_parse_action_variants():
    assert parse_action("ATTACK") == "ATTACK"
    assert parse_action("I would move left.") == "MOVE_LEFT"
    assert parse_action("move_right!") == "MOVE_RIGHT"


def test_parse_action_rejects_unclear():
    assert parse_action("MOVE_LEFT or ATTACK") is None
    assert parse_action("no idea") is None
    assert parse_action("") is None
    assert parse_action(None) is None


def test_policy_returns_action_and_counts_tokens():
    p, _, _ = make(["MOVE_LEFT"])
    assert p(STATE) == "MOVE_LEFT"
    assert p.stats.calls == 1
    assert p.stats.tokens_in == 100
    assert p.stats.tokens_out == 5


def test_prompt_contains_state_json():
    p, client, _ = make(["ATTACK"])
    p(STATE)
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert json.loads(msgs[-1]["content"]) == STATE


def test_retries_with_exponential_backoff():
    p, _, sleeps = make([Boom(), Boom(), "ATTACK"])
    assert p(STATE) == "ATTACK"
    assert sleeps == [2, 4]
    assert p.stats.retries == 2


def test_gives_up_after_max_retries_and_uses_fallback():
    p, _, _ = make([Boom(), Boom(), Boom()], max_retries=2, fallback="MOVE_LEFT")
    assert p(STATE) == "MOVE_LEFT"
    assert p.stats.failures == 1


def test_invalid_reply_uses_fallback_and_is_counted():
    p, _, _ = make(["I refuse"], fallback="MOVE_RIGHT")
    assert p(STATE) == "MOVE_RIGHT"
    assert p.stats.invalid == 1


def test_non_retryable_errors_propagate():
    p, _, _ = make([ValueError("bad request")])
    with pytest.raises(ValueError):
        p(STATE)


def test_cost_usd():
    s = LLMStats(tokens_in=1_000_000, tokens_out=500_000)
    assert cost_usd(s, 0.1, 0.4) == pytest.approx(0.3)

def test_retry_reasons_are_counted_by_error_type():
    p, _, _ = make([Boom(), Boom(), "ATTACK"])
    p(STATE)
    assert dict(p.stats.retry_reasons) == {"Boom": 2}