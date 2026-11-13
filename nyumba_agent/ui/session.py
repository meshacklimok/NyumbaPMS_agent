"""What the Streamlit page needs, kept free of Streamlit so tests can run it: pick the model client, build the
agent environment, and the scripted demo model used while there is no API key.
"""

import json
import os

from .. import config
from ..agent.prompts import build_system_prompt
from ..llm.base import LLMClient, Reply
from ..llm.fake import FakeClient, call, say
from ..tools.domain_tools import build_registry

DEMO_QUESTION = "Which property has the worst collection rate this year? Show a chart"

# First attempt uses a column that does not exist, so the trace shows an error and a retry (the loop's point).
_BAD_SQL = "SELECT p.name, ROUND(100.0 * SUM(v.collected) / SUM(v.total), 1) AS rate FROM v_invoice_state v JOIN property p ON p.id = v.property_id GROUP BY p.name"
_GOOD_SQL = ("SELECT p.name AS property, ROUND(100.0 * SUM(v.amount_paid) / SUM(v.total), 1) AS collection_rate "
             "FROM v_invoice_state v JOIN property p ON p.id = v.property_id "
             "WHERE v.period_start >= '{year}-01-01' AND v.period_start <= '{today}' "
             "GROUP BY p.name ORDER BY collection_rate")


def load_env(path=config.ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env (gitignored) into the environment, without overriding what is already set."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _answer_from_rows(messages: list[dict]) -> Reply:
    """Write the demo answer from the numbers the SQL tool actually returned, never from constants."""
    for m in reversed(messages):
        for block in m["content"] if isinstance(m["content"], list) else []:
            if block.get("type") == "tool_result" and not block.get("is_error"):
                data = json.loads(block["content"])
                if "rows" in data:
                    (name, rate), top = data["rows"][0], data["rows"][-1]
                    return say(f"**{name}** has the worst collection rate this year at {rate}% of billed rent "
                               f"(best: {top[0]} at {top[1]}%). The chart compares all properties, "
                               f"January to {config.TODAY:%B %Y}.")
    return say("I could not read the results.")


def demo_client(question: str) -> FakeClient:
    """A scripted stand-in for the model. It only knows the demo question; anything else gets an honest refusal."""
    if question.strip().rstrip("?.").lower() != DEMO_QUESTION.lower().rstrip("?."):
        return FakeClient([say("I am running in demo mode without a language model, so I can only answer: "
                               f"\"{DEMO_QUESTION}\". Set OPENAI_API_KEY and NYUMBA_LLM=openai to ask anything.")])
    sql = _GOOD_SQL.format(year=config.TODAY.year, today=config.TODAY.isoformat())
    return FakeClient([
        call("run_sql", "I will compute collected / billed per property for this year.", query=_BAD_SQL),
        call("run_sql", "That column does not exist; amount_paid is the right one.", query=sql),
        call("make_chart", "Now a chart of the same data.", type="bar", sql=sql, x="property", y="collection_rate",
             title=f"Collection rate by property, {config.TODAY.year} (%)"),
        _answer_from_rows,
    ])


def backend() -> str:
    """The one setting: NYUMBA_LLM=fake|openai. Unset means openai once OPENAI_API_KEY exists, else the demo."""
    return os.getenv("NYUMBA_LLM") or ("openai" if os.getenv("OPENAI_API_KEY") else "fake")


def make_client(backend: str, question: str) -> LLMClient:
    if backend == "fake":
        return demo_client(question)
    if backend == "openai":
        from ..llm.openai_client import OpenAIClient
        return OpenAIClient()
    raise ValueError(f"NYUMBA_LLM must be 'fake' or 'openai', not '{backend}'")


def build_env(db_path=None):
    return build_registry(db_path), build_system_prompt(db_path)
