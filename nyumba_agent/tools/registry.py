"""One shape for every tool: Tool(name, description, schema, run). The registry validates arguments and turns
every failure into {"error": ...} data, so the model can read it and retry (docs/02, principle 3).
"""

import json
from dataclasses import dataclass
from typing import Callable

import jsonschema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict  # JSON Schema for the arguments (the Anthropic `input_schema`)
    run: Callable[[dict], dict]


class Registry:
    def __init__(self, tools: list[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        """Tool definitions in the Anthropic Messages shape."""
        return [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in self._tools.values()]

    def call(self, name: str, args) -> dict:
        """Never raises. Returns the tool's dict, or {"error": ...}."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'. Available tools: {', '.join(self._tools)}"}
        if not isinstance(args, dict):
            return {"error": "tool arguments must be a JSON object"}
        try:
            jsonschema.validate(args, tool.schema)
        except jsonschema.ValidationError as exc:
            where = ".".join(map(str, exc.absolute_path))
            return {"error": f"invalid arguments{' for ' + where if where else ''}: {exc.message}"}
        try:
            result = tool.run(args)
        except Exception as exc:  # a bug in a tool must not kill the loop
            return {"error": f"tool '{name}' failed: {type(exc).__name__}: {exc}"}
        return result if isinstance(result, dict) else {"error": f"tool '{name}' returned a non-dict result"}


def model_view(result: dict, max_chars: int = 6000) -> str:
    """What the model gets to read: JSON without `_artifact` keys (figures), cut to a sensible length."""
    text = json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + f'... [cut: {len(text) - max_chars} more characters. Aggregate or select fewer columns]'
    return text
