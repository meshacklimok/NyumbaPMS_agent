"""Reminder guardrails: nothing is SENT without a human decision, no draft for a paid invoice, feedback is
saved, and repeated rejections raise a tenant's nudge threshold."""

import sqlite3

import pytest

from nyumba_agent import config, db, outbox
from nyumba_agent.data.generate import generate
from nyumba_agent.tools.domain_tools import build_registry
from nyumba_agent.tools.reminder_tool import invoice_facts


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / "nyumba.db"
    generate(path, seed=42)
    monkeypatch.setattr(config, "NYUMBA_DB", path)
    conn = outbox.open_outbox(db.connect_rw(tmp_path / "agent.db"))
    ro = db.connect_ro(path)
    overdue = [r["invoice_id"] for r in ro.execute("SELECT invoice_id FROM v_arrears ORDER BY invoice_id")]
    paid = ro.execute("SELECT invoice_id FROM v_invoice_state WHERE balance <= 0 LIMIT 1").fetchone()[0]
    ro.close()
    reg = build_registry(path, tmp_path / "agent.db")
    yield reg, conn, overdue, paid
    conn.close()


def draft(reg, invoice_id):
    r = reg.call("draft_reminder", {"invoice_id": invoice_id})
    assert "error" not in r, r
    return r["outbox_id"]


def test_draft_is_saved_as_draft_and_never_shows_the_phone(env):
    reg, conn, overdue, _ = env
    r = reg.call("draft_reminder", {"invoice_id": overdue[0]})
    assert r["status"] == "DRAFT" and r["amount"] in r["body"]
    assert "+254" not in str(r) and "07000" not in str(r)
    assert outbox.listing(conn)[0]["status"] == "DRAFT"


def test_language_follows_the_tenant(env):
    reg, conn, overdue, _ = env
    seen = {}
    for inv in overdue[:15]:
        r = reg.call("draft_reminder", {"invoice_id": inv})
        if "error" not in r:
            seen[r["language"]] = r["body"]
    assert "Habari" in seen["sw"] and "Hello" in seen["en"]


def test_no_draft_for_a_paid_invoice(env):
    reg, conn, _, paid = env
    assert "paid" in reg.call("draft_reminder", {"invoice_id": paid})["error"]
    assert outbox.listing(conn) == []


def test_no_second_draft_while_one_is_waiting(env):
    reg, conn, overdue, _ = env
    draft(reg, overdue[0])
    assert "already" in reg.call("draft_reminder", {"invoice_id": overdue[0]})["error"]


def test_draft_cannot_be_sent(env):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    with pytest.raises(outbox.OutboxError):
        outbox.send(conn, oid)
    assert outbox._row(conn, oid)["status"] == "DRAFT"


def test_rejected_message_cannot_be_sent(env):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    outbox.reject(conn, oid, "tenant called, paying tomorrow")
    with pytest.raises(outbox.OutboxError):
        outbox.send(conn, oid)


def test_database_itself_refuses_sent_without_approval(env):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    with pytest.raises(sqlite3.DatabaseError, match="without human approval"):
        conn.execute("UPDATE outbox SET status = 'SENT' WHERE id = ?", (oid,))


def test_approve_then_send_is_logged_and_saved(env, caplog):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    outbox.approve(conn, oid, invoice_facts(overdue[0]))
    with caplog.at_level("INFO", logger="nyumba.outbox"):
        line = outbox.send(conn, oid)
    assert "SIMULATED" in line and "SIMULATED" in caplog.text
    assert "07000" not in line  # the phone is masked even in the log
    row = outbox._row(conn, oid)
    assert row["status"] == "SENT" and row["sent_at"]
    assert [f["decision"] for f in conn.execute("SELECT decision FROM feedback")] == ["APPROVED"]


def test_edit_is_validated_and_sent_text_is_the_edit(env):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    facts = invoice_facts(overdue[0])
    with pytest.raises(outbox.OutboxError, match="amount"):
        outbox.edit(conn, oid, f"Pay now. Due {facts['due']}", facts)
    assert outbox._row(conn, oid)["status"] == "DRAFT"
    new = f"Please pay {facts['amount']} (due {facts['due']}). Thanks."
    outbox.edit(conn, oid, new, facts, "shorter")
    assert outbox._row(conn, oid)["status"] == "EDITED"
    assert new in outbox.send(conn, oid)
    fb = conn.execute("SELECT decision, edited, comment FROM feedback").fetchone()
    assert (fb["decision"], fb["edited"], fb["comment"]) == ("EDITED", 1, "shorter")


def test_a_decided_message_cannot_be_decided_again(env):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    outbox.reject(conn, oid)
    with pytest.raises(outbox.OutboxError):
        outbox.approve(conn, oid, invoice_facts(overdue[0]))


def test_sending_is_refused_if_the_invoice_got_paid_meanwhile(env, tmp_path):
    reg, conn, overdue, _ = env
    oid = draft(reg, overdue[0])
    outbox.approve(conn, oid, invoice_facts(overdue[0]))
    rw = sqlite3.connect(config.NYUMBA_DB)
    rw.execute("UPDATE invoice SET amount_paid = total, status = 'PAID' WHERE id = ?", (overdue[0],))
    rw.commit(); rw.close()
    with pytest.raises(outbox.OutboxError, match="already paid"):
        outbox.send(conn, oid)
    assert outbox._row(conn, oid)["status"] == "APPROVED"


def test_repeated_rejections_raise_the_nudge_threshold(env):
    reg, conn, overdue, _ = env
    tid = invoice_facts(overdue[0])["tenant_id"]
    assert outbox.nudge_threshold(conn, tid, 0.5) == 0.5
    for decision in ("REJECTED", "REJECTED", "APPROVED"):
        conn.execute("INSERT INTO feedback(outbox_id, tenant_id, decision, edited, created_at) VALUES (0,?,?,0,'x')",
                     (tid, decision))
    assert outbox.nudge_threshold(conn, tid, 0.5) == pytest.approx(0.6)
    conn.execute("INSERT INTO feedback(outbox_id, tenant_id, decision, edited, created_at) VALUES (0,?,'REJECTED',0,'x')", (tid,))
    assert outbox.nudge_threshold(conn, tid, 0.5) == pytest.approx(0.7)
    assert outbox.nudge_threshold(conn, tid + 1, 0.5) == 0.5  # other tenants are unaffected
