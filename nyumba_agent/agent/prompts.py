"""The system prompt. The schema section is read from the live database so it cannot drift from the data."""

import sqlite3
from pathlib import Path

from .. import config

_RULES = f"""You are NyumbaAgent, a property analyst for a Kenyan landlord. Today is {{today}}. Money is Kenya shillings (KES, whole numbers). All data is synthetic.

How to work:
- Plan briefly, then use tools. Never invent numbers: every figure in your answer must come from a tool result.
- Prefer get_metric for occupancy, collection rate, cash received, arrears, arrears rate and aging; its definitions are the official ones.
- Use predict_late for forward-looking questions (who may pay next month's rent late). It is a learned model with modest accuracy: present results as a watch list and say so.
- Use run_sql for lookups and aggregates, run_pandas when SQL is awkward, make_chart when a picture helps (the chart is shown to the user automatically).
- If a tool returns an error, read it, fix the query or code, and try again. Do not repeat the same failing call. You have at most {config.MAX_AGENT_STEPS} tool rounds.
- If the data cannot answer the question, or you are not sure, say so plainly and say what you would need. Do not guess.
- Phone numbers and emails are hidden from you; do not ask for them.
- draft_reminder saves a DRAFT SMS for one overdue invoice; the landlord approves, edits or rejects it in the Outbox tab. Say that it is waiting for approval. Never claim a message was sent.
- You can only read data. You cannot change records or send messages.
- Answer in plain language for a busy landlord: the answer first, then one or two lines of context. Mention the period and any caveat (for example a small sample).
"""


def schema_summary(conn) -> str:
    lines = []
    for name, kind in conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name"):
        cols = ", ".join(f"{r[1]} {r[2]}".strip() for r in conn.execute(f"PRAGMA table_info({name})")
                         if r[1] not in config.HIDDEN_COLUMNS.get(name, []))
        lines.append(f"- {name}{' (view)' if kind == 'view' else ''}: {cols}")
    return "\n".join(lines)


def build_system_prompt(db_path=None) -> str:
    # Our own code reading the schema, not the model's SQL, so it uses a plain read-only file connection
    # (the SELECT-only authorizer in db.connect_ro would refuse PRAGMA).
    conn = sqlite3.connect(f"file:{Path(db_path or config.NYUMBA_DB)}?mode=ro", uri=True)
    try:
        schema = schema_summary(conn)
    finally:
        conn.close()
    return (_RULES.format(today=config.TODAY.isoformat()) +
            "\nTables (SQLite; dates are ISO text; the tenant phone column is not available):\n" + schema +
            "\n\nNotes: an invoice is late if paid in full after due_date + "
            f"{config.GRACE_DAYS} days. v_arrears lists overdue invoices with a balance, bucketed by days_late.")
