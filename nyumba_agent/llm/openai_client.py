"""The OpenAI client: translates our `create(system, messages, tools) -> Reply` interface to Chat Completions.

Our messages are in the Anthropic Messages shape (llm/base.py), so each one is mapped to OpenAI's: assistant
`tool_use` blocks become `tool_calls`, and a user message of `tool_result` blocks becomes one `tool` message each.
The API key comes from the environment (OPENAI_API_KEY; `.env` is gitignored).
"""

import json

from .. import config
from .base import Reply, ToolCall


class LLMRefused(RuntimeError):
    """The reply cannot be used: it was cut off mid-answer or blocked by a content filter."""


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            msg = {"role": "assistant", "content": text or None}
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in content if b["type"] == "tool_use"]
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                else:
                    out.append({"role": "user", "content": b["text"]})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class OpenAIClient:
    def __init__(self, model: str = config.LLM_MODEL, max_tokens: int = config.LLM_MAX_TOKENS, sdk=None):
        if sdk is None:
            import openai  # imported here so the fake-client tests do not need the package or a key
            sdk = openai.OpenAI()
        self._sdk, self.model, self.max_tokens = sdk, model, max_tokens

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        params = dict(model=self.model, max_completion_tokens=self.max_tokens,
                      messages=to_openai_messages(system, messages))
        if tools:
            params["tools"] = to_openai_tools(tools)

        response = self._sdk.chat.completions.create(**params)
        choice = response.choices[0]
        if choice.finish_reason in ("length", "content_filter"):
            raise LLMRefused(f"finish_reason={choice.finish_reason}")

        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed_arguments": tc.function.arguments}  # the tool schema check turns this into an error the model can fix
            calls.append(ToolCall(tc.id, tc.function.name, args))
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        return Reply(text=choice.message.content or "", tool_calls=calls, usage=usage)
