"""End to end with no network: a scripted fake of the OpenAI SDK -> OpenAIClient -> the real agent loop, tools,
database and run log. If this passes, only the live call itself is untested."""

import json
import sqlite3
from types import SimpleNamespace as NS

import pytest

from nyumba_agent.agent.prompts import build_system_prompt
from nyumba_agent.agent.run_log import run_logged
from nyumba_agent.data.generate import generate
from nyumba_agent.llm.openai_client import OpenAIClient
from nyumba_agent.tools.domain_tools import build_registry


class FakeOpenAI:
    """Plays back scripted chat completions and records every request it was sent."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=self)

    def create(self, **params):
        self.requests.append(json.loads(json.dumps(params)))
        step = self.script.pop(0)
        return step(params) if callable(step) else step


def completion(text="", calls=(), finish="stop"):
    tcs = [NS(id=f"call_{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(calls)] or None
    return NS(choices=[NS(message=NS(content=text, tool_calls=tcs), finish_reason=finish)],
              usage=NS(prompt_tokens=100, completion_tokens=20))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(path, seed=42)
    return build_registry(path), build_system_prompt(path)


@pytest.fixture
def log():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_question_to_answer_with_a_retry_and_a_logged_run(env, log):
    reg, system = env
    sdk = FakeOpenAI([
        completion("Counting units.", [("run_sql", {"query": "SELECT COUNT(*) FROM unit_typo"})], "tool_calls"),
        completion("Wrong table; retrying.", [("run_sql", {"query": "SELECT COUNT(*) AS n FROM unit"})], "tool_calls"),
        completion("You have 120 units."),
    ])
    run = run_logged("How many units?", OpenAIClient(model="gpt-test", sdk=sdk), reg, system, conn=log)

    assert run.stop_reason == "answered" and run.answer == "You have 120 units." and run.retries == 1
    # the model was shown the SQL error as a tool message, in OpenAI's shape
    tool_msgs = [m for m in sdk.requests[1]["messages"] if m["role"] == "tool"]
    assert "no such table" in json.loads(tool_msgs[0]["content"])["error"]
    assert sdk.requests[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "run_sql"

    row = log.execute("SELECT * FROM run_log").fetchone()
    assert (row["question"], row["stop_reason"], row["model"], row["model_calls"]) == ("How many units?", "answered", "gpt-test", 3)
    assert (row["tokens_in"], row["tokens_out"]) == (300, 60) and row["seconds"] >= 0
    steps = json.loads(row["steps_json"])
    assert [s["is_error"] for s in steps] == [True, False] and steps[1]["result"]["rows"] == [[120]]


def test_step_limit_is_logged_and_the_last_call_offers_no_tools(env, log):
    reg, system = env
    again = completion("", [("run_sql", {"query": "SELECT 1"})], "tool_calls")
    sdk = FakeOpenAI([again] * 3)
    run = run_logged("loop forever", OpenAIClient(sdk=sdk), reg, system, conn=log, max_steps=2)
    assert run.stop_reason == "step_limit" and "tools" not in sdk.requests[2]
    assert log.execute("SELECT stop_reason FROM run_log").fetchone()[0] == "step_limit"


def test_api_failure_is_logged_not_raised(env, log):
    reg, system = env

    def boom(params):
        raise ConnectionError("no network")

    run = run_logged("q", OpenAIClient(sdk=FakeOpenAI([boom])), reg, system, conn=log)
    assert run.stop_reason == "model_error" and "ConnectionError" in run.answer
    assert log.execute("SELECT stop_reason FROM run_log").fetchone()[0] == "model_error"
