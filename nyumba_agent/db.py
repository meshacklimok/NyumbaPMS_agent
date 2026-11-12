"""Database access. Every analysis tool goes through `run_select`: read-only file, SELECT-only authorizer,
row cap and timeout (docs/02 guardrails). Nothing here returns an exception; tools hand errors back as data
so the model can read them and retry.
"""

import re
import sqlite3
import time
from pathlib import Path

from . import config

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorizer(action, arg1, arg2, db_name, source):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


# Contact details never reach the model through SQL (docs/02 privacy). Drafting a reminder goes through
# its own tool, which reads the phone number itself.
_MASKS = (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), re.compile(r"\+?254[\s-]?\d{9}\b"),
          re.compile(r"\b0[17]\d{8}\b"))


def mask(value):
    if not isinstance(value, str):
        return value
    for pattern in _MASKS:
        value = pattern.sub("[hidden]", value)
    return value


def connect_ro(path: Path | str | None = None) -> sqlite3.Connection:
    """A connection that cannot write: `mode=ro` at the file level, plus a SELECT-only authorizer."""
    path = Path(path or config.NYUMBA_DB)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_authorizer)
    return conn


def connect_rw(path: Path | str | None = None) -> sqlite3.Connection:
    """For the agent's own database (outbox, feedback, logs) and for the data generator only."""
    path = Path(path or config.AGENT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def run_select(query: str, conn: sqlite3.Connection | None = None, limit: int = config.SQL_ROW_LIMIT,
               timeout: float = config.SQL_TIMEOUT_S) -> dict:
    """Run one SELECT. Returns {columns, rows, row_count, truncated} or {error}."""
    query = (query or "").strip().rstrip(";").strip()
    if not query:
        return {"error": "empty query"}
    if ";" in query:
        return {"error": "only one statement is allowed"}
    if query.split(None, 1)[0].upper() not in ("SELECT", "WITH"):
        return {"error": "only SELECT queries are allowed"}
    own = conn is None
    conn = conn or connect_ro()
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(limit + 1)
        columns = [d[0] for d in cur.description or []]
        truncated = len(rows) > limit
        return {"columns": columns, "rows": [[mask(v) for v in r] for r in rows[:limit]], "row_count": min(len(rows), limit),
                "truncated": truncated}
    except sqlite3.Error as exc:
        msg = str(exc)
        if "interrupted" in msg:
            msg = f"query took longer than {timeout:g}s and was stopped"
        elif "not authorized" in msg:
            msg = "not allowed: read-only SELECT queries only"
        return {"error": msg}
    finally:
        conn.set_progress_handler(None, 0)
        if own:
            conn.close()
