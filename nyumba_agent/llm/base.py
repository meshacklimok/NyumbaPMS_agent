"""The model interface, in the Anthropic Messages shape (D-001). A client takes the system prompt, the message
list and the tool specs, and returns one `Reply`. Real clients (Anthropic, Ollama) and `FakeClient` all fit it,
so the loop never knows which one it talks to.

Messages are plain dicts:
  {"role": "user", "content": "text"}
  {"role": "assistant", "content": [{"type": "text", "text": ...}, {"type": "tool_use", "id", "name", "input"}]}
  {"role": "user", "content": [{"type": "tool_result", "tool_use_id", "content": "json text", "is_error": bool}]}
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {"input_tokens", "output_tokens"} when the provider reports them

    def to_message(self) -> dict:
        content = ([{"type": "text", "text": self.text}] if self.text else []) + [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in self.tool_calls]
        return {"role": "assistant", "content": content}


class LLMClient(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Reply: ...
