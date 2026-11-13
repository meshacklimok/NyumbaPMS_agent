"""The outbox: every drafted reminder, its status and the landlord's decision on it (docs/03, docs/04).

DRAFT -> APPROVED | EDITED | REJECTED, and only APPROVED or EDITED -> SENT. A trigger enforces that last rule in
the database itself, so no code path (or stray UPDATE) can mark a message sent without a human decision.
Sending is simulated: it is logged and nothing leaves the machine. The phone number is read here, at send time;
it is never stored in the outbox and never reaches the model.
"""

import logging
import re
import sqlite3
from datetime import datetime, timezone

from . import config, db

log = logging.getLogger("nyumba.outbox")

STATUSES = ("DRAFT", "APPROVED", "EDITED", "REJECTED", "SENT")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox(
    id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL, invoice_id INTEGER NOT NULL,
    channel TEXT NOT NULL DEFAULT 'sms', language TEXT NOT NULL, body TEXT NOT NULL, edited_body TEXT,
    status TEXT NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT','APPROVED','EDITED','REJECTED','SENT')),
    created_at TEXT NOT NULL, decided_at TEXT, sent_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY, outbox_id INTEGER NOT NULL REFERENCES outbox(id), tenant_id INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('APPROVED','EDITED','REJECTED')), edited INTEGER NOT NULL,
    comment TEXT, created_at TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS outbox_send_needs_approval BEFORE UPDATE OF status ON outbox
WHEN NEW.status = 'SENT' AND OLD.status NOT IN ('APPROVED','EDITED')
BEGIN SELECT RAISE(ABORT, 'a message cannot be sent without human approval'); END;
"""


class OutboxError(Exception):
    """A refused action. The message is meant to be shown to the landlord."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_outbox(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    conn = conn or db.connect_rw()
    conn.executescript(_SCHEMA)
    return conn


def validate_body(body: str, facts: dict) -> None:
    """The text must state the real balance and due date, so a wrong number never gets approved."""
    if not (body or "").strip():
        raise OutboxError("the message is empty")
    if facts["amount"] not in body:
        raise OutboxError(f"the message must state the amount owed ({facts['amount']})")
    if facts["due"] not in body:
        raise OutboxError(f"the message must state the due date ({facts['due']})")


def add_draft(conn, tenant_id: int, invoice_id: int, language: str, body: str, facts: dict) -> int:
    open_outbox(conn)
    validate_body(body, facts)
    if conn.execute("SELECT 1 FROM outbox WHERE invoice_id = ? AND status IN ('DRAFT','APPROVED','EDITED')",
                    (invoice_id,)).fetchone():
        raise OutboxError(f"invoice {invoice_id} already has a reminder waiting in the outbox")
    cur = conn.execute("INSERT INTO outbox(tenant_id, invoice_id, language, body, created_at) VALUES (?,?,?,?,?)",
                       (tenant_id, invoice_id, language, body, _now()))
    conn.commit()
    return cur.lastrowid


def _row(conn, outbox_id: int):
    row = conn.execute("SELECT * FROM outbox WHERE id = ?", (outbox_id,)).fetchone()
    if row is None:
        raise OutboxError(f"no outbox message {outbox_id}")
    return row


def _decide(conn, outbox_id: int, status: str, facts: dict | None = None, body: str | None = None,
            comment: str | None = None) -> None:
    row = _row(conn, outbox_id)
    if row["status"] != "DRAFT":
        raise OutboxError(f"message {outbox_id} is {row['status']}, only a DRAFT can be decided")
    edited = status == "EDITED"
    if status != "REJECTED":
        validate_body(body if edited else row["body"], facts)
    now = _now()
    conn.execute("UPDATE outbox SET status = ?, edited_body = ?, decided_at = ?, reason = ? WHERE id = ?",
                 (status, body if edited else None, now, comment, outbox_id))
    conn.execute("INSERT INTO feedback(outbox_id, tenant_id, decision, edited, comment, created_at) VALUES (?,?,?,?,?,?)",
                 (outbox_id, row["tenant_id"], status, int(edited), comment, now))
    conn.commit()


def approve(conn, outbox_id: int, facts: dict) -> None:
    _decide(conn, outbox_id, "APPROVED", facts)


def edit(conn, outbox_id: int, new_body: str, facts: dict, comment: str | None = None) -> None:
    _decide(conn, outbox_id, "EDITED", facts, new_body, comment)


def reject(conn, outbox_id: int, reason: str | None = None) -> None:
    _decide(conn, outbox_id, "REJECTED", comment=reason)


def send(conn, outbox_id: int, nyumba_db=None) -> str:
    """Simulated send of an approved message. Re-checks the invoice first: a tenant who paid since the draft
    was approved must not be nagged. Returns the log line."""
    row = _row(conn, outbox_id)
    if row["status"] not in ("APPROVED", "EDITED"):
        raise OutboxError(f"message {outbox_id} is {row['status']}: only an approved message can be sent")
    src = db.connect_ro(nyumba_db)
    try:
        inv = src.execute("SELECT balance FROM v_invoice_state WHERE invoice_id = ?", (row["invoice_id"],)).fetchone()
        phone = src.execute("SELECT phone FROM tenant WHERE id = ?", (row["tenant_id"],)).fetchone()["phone"]
    finally:
        src.close()
    if inv is None or inv["balance"] <= 0:
        raise OutboxError(f"invoice {row['invoice_id']} is already paid, so the reminder was not sent")
    text = row["edited_body"] or row["body"]
    line = f"[SIMULATED SMS] to {phone[:6]}***{phone[-2:]}: {text}"
    log.info(line)
    conn.execute("UPDATE outbox SET status = 'SENT', sent_at = ? WHERE id = ?", (_now(), outbox_id))
    conn.commit()
    return line


def listing(conn, status: str | None = None) -> list[sqlite3.Row]:
    open_outbox(conn)
    q = "SELECT * FROM outbox" + (" WHERE status = ?" if status else "") + " ORDER BY id DESC"
    return conn.execute(q, (status,) if status else ()).fetchall()


# --- learning from feedback ---------------------------------------------------------------------------

FEEDBACK_WINDOW = 5  # a tenant's most recent decisions
MIN_DECISIONS = 3  # below this there is too little evidence to change anything
THRESHOLD_STEP = 0.10
THRESHOLD_CAP = 0.95


def nudge_threshold(conn, tenant_id: int, base: float) -> float:
    """Raise the 'send a nudge' probability threshold for a tenant whose drafts the landlord keeps rejecting:
    +0.10 when at least half of their last five decisions were rejections, +0.20 from three quarters."""
    rows = conn.execute("SELECT decision FROM feedback WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
                        (tenant_id, FEEDBACK_WINDOW)).fetchall()
    if len(rows) < MIN_DECISIONS:
        return base
    rate = sum(r["decision"] == "REJECTED" for r in rows) / len(rows)
    steps = 2 if rate >= 0.75 else 1 if rate >= 0.5 else 0
    return min(base + steps * THRESHOLD_STEP, THRESHOLD_CAP)


def nudge_thresholds(conn, tenant_ids, base: float) -> dict[int, float]:
    open_outbox(conn)
    return {t: nudge_threshold(conn, t, base) for t in tenant_ids}
