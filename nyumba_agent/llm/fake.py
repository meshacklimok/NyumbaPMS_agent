"""A scripted model for tests: no network, no cost. Each call to `create` plays back the next scripted step.

A step is a `Reply`, or a function `(messages) -> Reply` when the reply must depend on what the loop sent
(for example, reacting to a tool error). Every call is recorded in `calls` so tests can inspect what the
model was shown.
"""

import copy
import itertools

from .base import Reply, ToolCall

_ids = itertools.count(1)


def say(text: str) -> Reply:
    return Reply(text=text)


def call(name: str, text: str = "", **tool_input) -> Reply:
    return Reply(text=text, tool_calls=[ToolCall(f"toolu_{next(_ids)}", name, tool_input)])


class ScriptExhausted(AssertionError):
    pass


class FakeClient:
    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[dict] = []

    def create(self, system, messages, tools) -> Reply:
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self._script:
            raise ScriptExhausted(f"the agent asked the model a {len(self.calls)}th time but the script has no more steps")
        step = self._script.pop(0)
        return step(messages) if callable(step) else step

    @property
    def remaining(self) -> int:
        return len(self._script)
