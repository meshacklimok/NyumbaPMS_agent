"""The OpenAI client against a stubbed SDK object: checks the translation, never touches the network."""

import json
from types import SimpleNamespace as NS

import pytest

from nyumba_agent.llm.openai_client import LLMRefused, OpenAIClient

TOOLS = [{"name": "run_sql", "description": "d", "input_schema": {"type": "object"}}]


class StubSDK:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.chat = NS(completions=self)

    def create(self, **params):
        self.requests.append(params)
        return self.responses.pop(0)


def response(content="", tool_calls=None, finish="stop"):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason=finish)], usage=NS(prompt_tokens=10, completion_tokens=5))


def tc(id, name, arguments):
    return NS(id=id, function=NS(name=name, arguments=arguments))


def test_reply_translation():
    sdk = StubSDK(response("Checking.", [tc("call_1", "run_sql", '{"query": "select 1"}')], "tool_calls"))
    reply = OpenAIClient(sdk=sdk).create("sys", [{"role": "user", "content": "q"}], TOOLS)
    assert reply.text == "Checking."
    assert [(c.id, c.name, c.input) for c in reply.tool_calls] == [("call_1", "run_sql", {"query": "select 1"})]
    assert reply.usage == {"input_tokens": 10, "output_tokens": 5}


def test_history_and_tools_are_translated():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "let me look"},
                                          {"type": "tool_use", "id": "c1", "name": "run_sql", "input": {"query": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{\"rows\": []}",
                                      "is_error": False}]},
    ]
    sdk = StubSDK(response("done"))
    OpenAIClient(model="m", max_tokens=99, sdk=sdk).create("sys", history, TOOLS)
    req = sdk.requests[0]
    assert req["model"] == "m" and req["max_completion_tokens"] == 99
    assert req["tools"] == [{"type": "function", "function": {"name": "run_sql", "description": "d",
                                                              "parameters": {"type": "object"}}}]
    sys_, user, asst, tool = req["messages"]
    assert sys_ == {"role": "system", "content": "sys"} and user["content"] == "q"
    assert asst["content"] == "let me look"
    assert asst["tool_calls"] == [{"id": "c1", "type": "function",
                                   "function": {"name": "run_sql", "arguments": json.dumps({"query": "x"})}}]
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "{\"rows\": []}"}


def test_no_tools_sends_no_tools_param():
    sdk = StubSDK(response("a"))
    OpenAIClient(sdk=sdk).create("s", [{"role": "user", "content": "q"}], [])
    assert "tools" not in sdk.requests[0]


def test_bad_arguments_json_becomes_an_input_the_schema_check_can_reject():
    sdk = StubSDK(response("", [tc("c", "run_sql", "{oops")], "tool_calls"))
    reply = OpenAIClient(sdk=sdk).create("s", [], TOOLS)
    assert reply.tool_calls[0].input == {"_unparsed_arguments": "{oops"}


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_unusable_stops_raise(finish):
    with pytest.raises(LLMRefused):
        OpenAIClient(sdk=StubSDK(response("par", finish=finish))).create("s", [], TOOLS)
