"""The whole plan -> tool -> observe -> retry cycle, driven by a scripted model: no network, no cost."""

import json

import pytest

from nyumba_agent.agent.loop import run_agent
from nyumba_agent.agent.prompts import build_system_prompt
from nyumba_agent.data.generate import generate
from nyumba_agent.llm.fake import FakeClient, ScriptExhausted, call, say
from nyumba_agent.tools.domain_tools import build_registry


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(path, seed=42)
    return build_registry(path), build_system_prompt(path)


def ask(env, script, question="q", **kw):
    reg, system = env
    client = FakeClient(script)
    return run_agent(question, client, reg, system, **kw), client


def tool_results(client, call_no):
    """The tool_result blocks the model was shown on its `call_no`th call (0-based)."""
    return client.calls[call_no]["messages"][-1]["content"]


def test_answers_without_tools(env):
    run, client = ask(env, [say("Hello.")])
    assert run.answer == "Hello." and run.stop_reason == "answered" and run.steps == []
    assert client.calls[0]["tools"]  # the model was offered the tools


def test_plan_tool_observe_answer(env):
    run, client = ask(env, [
        call("run_sql", "I will count the units.", query="SELECT COUNT(*) AS n FROM unit"),
        say("You have 120 units."),
    ])
    assert run.stop_reason == "answered" and run.answer == "You have 120 units."
    assert [s.tool for s in run.steps] == ["run_sql"] and not run.steps[0].is_error
    seen = tool_results(client, 1)[0]
    assert seen["type"] == "tool_result" and json.loads(seen["content"])["rows"] == [[120]] and not seen["is_error"]


def test_recovers_from_a_bad_column_name(env):
    def fix_the_query(messages):
        error = json.loads(messages[-1]["content"][0]["content"])["error"]
        assert "no such column: rent_amount" in error  # the model really was shown the error
        return call("run_sql", "That column does not exist; the table has list_rent.",
                    query="SELECT SUM(list_rent) AS total FROM unit")

    run, client = ask(env, [
        call("run_sql", query="SELECT SUM(rent_amount) FROM unit"),
        fix_the_query,
        say("Listed rent adds up to the amount shown."),
    ])
    assert run.stop_reason == "answered" and [s.is_error for s in run.steps] == [True, False]
    assert run.retries == 1
    assert tool_results(client, 1)[0]["is_error"] is True
    assert tool_results(client, 2)[0]["is_error"] is False


def test_recovers_from_a_sandbox_error(env):
    run, _ = ask(env, [
        call("run_pandas", code="result = unit['rent'].sum()"),
        call("run_pandas", code="result = int(unit['list_rent'].sum())"),
        say("done"),
    ])
    assert [s.is_error for s in run.steps] == [True, False] and run.steps[1].result["value"] > 0


def test_unknown_tool_and_bad_arguments_are_observations(env):
    run, _ = ask(env, [call("delete_everything"), call("run_sql"), say("I cannot do that.")])
    assert run.stop_reason == "answered" and all(s.is_error for s in run.steps)
    assert "unknown tool" in run.steps[0].result["error"] and "invalid arguments" in run.steps[1].result["error"]


def test_chart_goes_to_artifacts_and_not_to_the_model(env):
    run, client = ask(env, [
        call("make_chart", type="bar", title="Units by status", x="status", y="n",
             sql="SELECT status, COUNT(*) AS n FROM unit GROUP BY status"),
        say("Here is the chart."),
    ])
    assert len(run.artifacts) == 1 and run.artifacts[0]["kind"] == "chart"
    assert json.loads(run.artifacts[0]["figure"])["data"]
    assert "plotly" not in json.dumps(client.calls[1]["messages"]) and "_figure" not in json.dumps(client.calls[1]["messages"])


def test_step_limit_stops_a_looping_model(env):
    always_sql = [call("run_sql", query="SELECT 1") for _ in range(20)]
    run, client = ask(env, always_sql, max_steps=3)
    assert run.stop_reason == "step_limit" and len(run.steps) == 3
    assert len(client.calls) == 4 and client.calls[-1]["tools"] == []  # last call gets no tools
    assert "not giving a number" in run.answer


def test_model_gets_one_last_chance_to_answer(env):
    run, _ = ask(env, [call("run_sql", query="SELECT 1"), call("run_sql", query="SELECT 2"), say("It is 2.")], max_steps=2)
    assert run.stop_reason == "answered" and run.answer == "It is 2."


def test_model_failure_is_reported_not_raised(env):
    def boom(messages):
        raise ConnectionError("network down")
    run, _ = ask(env, [boom])
    assert run.stop_reason == "model_error" and "network down" in run.answer


def test_empty_answer_is_not_passed_off_as_success(env):
    run, _ = ask(env, [say("   ")])
    assert run.stop_reason == "model_error"


def test_script_exhaustion_is_loud(env):
    reg, system = env
    with pytest.raises(ScriptExhausted):
        FakeClient([]).create(system, [], [])


def test_history_is_kept_and_usage_counted(env):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
    run, client = ask(env, [say("ok")], history=history)
    assert len(client.calls[0]["messages"]) == 3 and run.usage["model_calls"] == 1


def test_system_prompt_describes_live_schema_without_contact_columns(env):
    _, system = env
    assert "invoice:" in system and "v_arrears (view)" in system
    assert "phone TEXT" not in system and "tenant: id INTEGER, name TEXT, language" in system
