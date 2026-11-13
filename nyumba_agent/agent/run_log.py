"""The run log: every AgentRun is written to agent.db (docs/03). It feeds the cost and latency table in the
evaluation. Steps are stored the way the model saw them (no figure payloads), so a run can be replayed by reading.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone

from .. import config
from ..db import connect_rw
from ..llm.base import LLMClient
from ..tools.registry import Registry, model_view
from .loop import AgentRun, run_agent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_log(
    id INTEGER PRIMARY KEY, question TEXT NOT NULL, steps_json TEXT NOT NULL, answer TEXT NOT NULL,
    stop_reason TEXT NOT NULL, model TEXT NOT NULL, model_calls INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL, seconds REAL NOT NULL, created_at TEXT NOT NULL)
"""


def log_run(run: AgentRun, seconds: float, model: str, conn: sqlite3.Connection | None = None) -> int:
    own = conn is None
    conn = conn or connect_rw()
    try:
        conn.execute(_SCHEMA)
        steps = [{"tool": s.tool, "input": s.input, "is_error": s.is_error, "result": json.loads(model_view(s.result))}
                 for s in run.steps]
        cur = conn.execute(
            "INSERT INTO run_log(question, steps_json, answer, stop_reason, model, model_calls, tokens_in, tokens_out,"
            " seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run.question, json.dumps(steps, default=str), run.answer, run.stop_reason, model,
             run.usage["model_calls"], run.usage["input_tokens"], run.usage["output_tokens"], round(seconds, 3),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def run_logged(question: str, client: LLMClient, registry: Registry, system: str, conn: sqlite3.Connection | None = None,
               model: str = config.LLM_MODEL, **kw) -> AgentRun:
    """`run_agent` plus a row in the run log."""
    start = time.monotonic()
    run = run_agent(question, client, registry, system, **kw)
    log_run(run, time.monotonic() - start, getattr(client, "model", model), conn)
    return run
