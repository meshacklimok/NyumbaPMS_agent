"""The agent loop, written by hand (D-003): plan -> call tool -> observe -> retry or answer.

Each round the model sees the conversation so far and either answers in text (done) or asks for tools. Tool
results, including errors, go back as tool_result blocks, so a bad column name is just one more observation
the model can correct. The loop is bounded by `max_steps` and never raises: failures end the run with a
`stop_reason` and an honest message (guardrail: say when unsure).
"""

from dataclasses import dataclass, field

from .. import config
from ..llm.base import LLMClient
from ..tools.registry import Registry, model_view


@dataclass
class Step:
    tool: str
    input: dict
    result: dict  # full result, including hidden `_` artifacts
    is_error: bool


@dataclass
class AgentRun:
    question: str
    answer: str
    stop_reason: str  # answered | step_limit | model_error
    steps: list[Step] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)  # [{"kind": "chart", "figure": <plotly json>, "title": ...}]
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "model_calls": 0})

    @property
    def retries(self) -> int:
        """Errors the model recovered from: error steps followed by a later success."""
        return sum(1 for i, s in enumerate(self.steps) if s.is_error and any(not t.is_error for t in self.steps[i + 1:]))


def run_agent(question: str, client: LLMClient, registry: Registry, system: str,
              history: list[dict] | None = None, max_steps: int = config.MAX_AGENT_STEPS) -> AgentRun:
    """`history` is earlier messages of the same conversation (already in message form)."""
    messages = list(history or []) + [{"role": "user", "content": question}]
    run = AgentRun(question=question, answer="", stop_reason="step_limit", messages=messages)
    tools = registry.specs()

    for round_no in range(max_steps + 1):  # max_steps tool rounds, then one last call without tools: it must answer
        last_chance = round_no == max_steps
        try:
            reply = client.create(system, messages, [] if last_chance else tools)
        except Exception as exc:
            run.stop_reason = "model_error"
            run.answer = f"I could not reach the language model ({type(exc).__name__}: {exc}). Please try again."
            return run
        run.usage["model_calls"] += 1
        for k in ("input_tokens", "output_tokens"):
            run.usage[k] += reply.usage.get(k, 0)
        messages.append(reply.to_message())

        if not reply.tool_calls:
            run.answer, run.stop_reason = reply.text.strip(), "answered"
            if not run.answer:
                run.answer, run.stop_reason = "The model returned an empty answer. Please rephrase and try again.", "model_error"
            return run
        if last_chance:
            break  # it still asked for tools after its last allowed round

        results = []
        for tc in reply.tool_calls:
            result = registry.call(tc.name, tc.input)
            is_error = "error" in result
            run.steps.append(Step(tc.name, tc.input, result, is_error))
            if "_figure" in result:
                run.artifacts.append({"kind": "chart", "figure": result["_figure"], "title": result.get("title", "")})
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": model_view(result),
                            "is_error": is_error})
        messages.append({"role": "user", "content": results})

    run.stop_reason = "step_limit"
    run.answer = (f"I could not finish within {max_steps} tool steps, so I am not giving a number I am unsure of. "
                  "Try a narrower question, or ask for one part at a time.")
    return run
