"""draft_reminder: turn one overdue invoice into a polite SMS in the tenant's language and put it in the outbox.

It writes a draft and nothing else; it has no way to send. The tenant's phone number is never read here, so it
cannot appear in the result the model sees. The text comes from fixed templates filled with database values, so
the amount and due date cannot be mistyped by a language model.
"""

from datetime import date

from .. import config, db, outbox
from .registry import Tool

# tone by days late: friendly, then firmer (docs/04)
_SW = {
    "soft": "Habari {first}, ni ukumbusho wa upole: kodi ya nyumba {unit} ya {amount} ilitakiwa kulipwa tarehe {due}. "
            "Tafadhali lipa kupitia M-Pesa Paybill {paybill}, akaunti {unit}. Asante!",
    "firm": "Habari {first}, kodi ya nyumba {unit} ya {amount} imechelewa siku {days} tangu tarehe {due}. "
            "Tafadhali lipa leo kupitia M-Pesa Paybill {paybill}, akaunti {unit}, au utupigie simu kama una shida.",
    "final": "Habari {first}, kodi ya nyumba {unit} ya {amount} iliyotakiwa tarehe {due} bado haijalipwa (siku {days}). "
             "Tafadhali lipa Paybill {paybill}, akaunti {unit}, au wasiliana nasi leo ili tukubaliane mpango wa malipo.",
}
_EN = {
    "soft": "Hello {first}, a gentle reminder: rent for {unit} of {amount} was due on {due}. Please pay via "
            "M-Pesa Paybill {paybill}, account {unit}. Thank you!",
    "firm": "Hello {first}, rent for {unit} of {amount}, due on {due}, is now {days} days overdue. Please pay today via "
            "M-Pesa Paybill {paybill}, account {unit}, or call us if you have a problem.",
    "final": "Hello {first}, rent for {unit} of {amount}, due on {due}, is still unpaid ({days} days). Please pay via "
             "Paybill {paybill}, account {unit}, or contact us today so we can agree a payment plan.",
}
_TEMPLATES = {"en": _EN, "sw": _SW}


def _tone(days_late: int) -> str:
    return "soft" if days_late <= 7 else "firm" if days_late <= 30 else "final"


def money(amount: int) -> str:
    return f"KES {amount:,}"


def long_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def invoice_facts(invoice_id: int, nyumba_db=None) -> dict | None:
    """What a reminder must say, from the database. No phone number."""
    conn = db.connect_ro(nyumba_db)
    try:
        row = conn.execute(
            "SELECT v.invoice_id, v.tenant_id, v.balance, v.due_date, v.days_late, v.status, t.name, t.language, "
            "u.code AS unit FROM v_invoice_state v JOIN tenant t ON t.id = v.tenant_id JOIN unit u ON u.id = v.unit_id "
            "WHERE v.invoice_id = ?", (invoice_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {**dict(row), "amount": money(row["balance"]), "due": long_date(row["due_date"])}


def compose(facts: dict) -> str:
    lang = facts["language"] if facts["language"] in _TEMPLATES else "en"
    return _TEMPLATES[lang][_tone(facts["days_late"])].format(
        first=facts["name"].split()[0], unit=facts["unit"], amount=facts["amount"], due=facts["due"],
        days=facts["days_late"], paybill=config.MPESA_PAYBILL)


def make_draft_reminder(nyumba_db=None, agent_db=None) -> Tool:
    def run(args):
        facts = invoice_facts(args["invoice_id"], nyumba_db)
        if facts is None:
            return {"error": f"no invoice {args['invoice_id']}"}
        if facts["balance"] <= 0 or facts["status"] == "PAID":
            return {"error": f"invoice {args['invoice_id']} is paid; no reminder is needed"}
        if facts["due_date"] >= config.TODAY.isoformat():
            return {"error": f"invoice {args['invoice_id']} is not overdue yet (due {facts['due_date']})"}
        conn = outbox.open_outbox(db.connect_rw(agent_db))
        try:
            body = compose(facts)
            try:
                outbox_id = outbox.add_draft(conn, facts["tenant_id"], facts["invoice_id"], facts["language"], body, facts)
            except outbox.OutboxError as exc:
                return {"error": str(exc)}
        finally:
            conn.close()
        return {"outbox_id": outbox_id, "status": "DRAFT", "tenant": facts["name"], "language": facts["language"],
                "amount": facts["amount"], "days_late": facts["days_late"], "body": body,
                "note": "Saved as a draft. Nothing has been sent: the landlord must approve it in the Outbox tab."}

    return Tool("draft_reminder",
                "Draft a polite rent-reminder SMS for ONE overdue, unpaid invoice, in the tenant's own language "
                "(English or Swahili). It only saves a DRAFT for the landlord to approve, edit or reject in the "
                "Outbox tab; it cannot send anything. Get invoice ids from v_arrears (invoice_id).",
                {"type": "object", "properties": {"invoice_id": {"type": "integer"}}, "required": ["invoice_id"],
                 "additionalProperties": False}, run)
