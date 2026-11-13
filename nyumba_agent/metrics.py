"""The business metrics, defined once (docs/03). The agent's domain tools, the manager jobs and the evaluation
reference answers all call these, so "arrears rate" means the same thing everywhere.

All functions take a connection from `db.connect_ro` and return plain numbers or dicts. Counts are returned
beside rates because a small portfolio makes percentages swing (docs/15 section 2.9).
"""

import datetime as dt
import sqlite3


def _one(conn: sqlite3.Connection, sql: str, *args):
    return conn.execute(sql, args).fetchone()[0]


def month_start(d: dt.date) -> str:
    return d.replace(day=1).isoformat()


def occupancy(conn) -> dict:
    total, occupied = conn.execute("SELECT COUNT(*), COALESCE(SUM(status='OCCUPIED'),0) FROM unit").fetchone()
    return {"occupied": occupied, "units": total, "rate": occupied / total if total else None}


def expected_rent(conn, month: dt.date, property_id: int | None = None) -> int:
    sql = ("SELECT COALESCE(SUM(total),0) FROM v_invoice_state WHERE period_start = ?"
           + (" AND property_id = ?" if property_id else ""))
    return _one(conn, sql, month_start(month), *([property_id] if property_id else []))


def collection_rate(conn, month: dt.date, property_id: int | None = None) -> dict:
    """Accrual view: of what was invoiced for the month, how much has been paid (so far)."""
    where = "period_start = ?" + (" AND property_id = ?" if property_id else "")
    args = [month_start(month)] + ([property_id] if property_id else [])
    invoices, paid_invoices, expected, collected = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(balance <= 0),0), COALESCE(SUM(total),0), COALESCE(SUM(amount_paid),0) "
        f"FROM v_invoice_state WHERE {where}", args).fetchone()
    return {"invoices": invoices, "paid_invoices": paid_invoices, "expected": expected, "collected": collected,
            "rate": collected / expected if expected else None}


def cash_received(conn, month: dt.date) -> int:
    start = month.replace(day=1)
    end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return _one(conn, "SELECT COALESCE(SUM(amount),0) FROM payment WHERE paid_on >= ? AND paid_on < ?",
                start.isoformat(), end.isoformat())


def arrears(conn, property_id: int | None = None) -> dict:
    """Everything overdue with a balance, as of the date baked into the views."""
    where = " WHERE property_id = ?" if property_id else ""
    args = [property_id] if property_id else []
    invoices, tenants, amount = conn.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT tenant_id), COALESCE(SUM(balance),0) FROM v_arrears{where}", args).fetchone()
    return {"invoices": invoices, "tenants": tenants, "amount": amount}


def arrears_rate(conn, month: dt.date, property_id: int | None = None) -> dict:
    """The goal metric: total arrears / this month's expected rent. Target is below 10%."""
    a, exp = arrears(conn, property_id), expected_rent(conn, month, property_id)
    return {**a, "expected_rent": exp, "rate": a["amount"] / exp if exp else None}


def aging_buckets(conn, property_id: int | None = None) -> dict:
    where = " WHERE property_id = ?" if property_id else ""
    rows = conn.execute(f"SELECT bucket, COUNT(*), SUM(balance) FROM v_arrears{where} GROUP BY bucket",
                        [property_id] if property_id else []).fetchall()
    found = {r[0]: {"invoices": r[1], "amount": r[2]} for r in rows}
    return {b: found.get(b, {"invoices": 0, "amount": 0}) for b in ("1-7", "8-30", "31-60", "60+")}
