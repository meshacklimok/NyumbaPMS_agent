"""Metrics checked against a tiny database whose answers were worked out by hand (docs/15 section 7)."""

import datetime as dt

import pytest

from nyumba_agent import db, metrics
from nyumba_agent.data.generate import create_schema

TODAY = dt.date(2026, 3, 20)
MARCH = dt.date(2026, 3, 1)


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "tiny.db"
    w = db.connect_rw(path)
    create_schema(w, TODAY)
    w.executescript("""
    INSERT INTO property VALUES (1,'P','Nairobi','Kasarani','APARTMENT_BLOCK');
    INSERT INTO unit VALUES (1,1,'A01','ONE_BEDROOM',1,10000,'OCCUPIED'),(2,1,'A02','TWO_BEDROOM',2,20000,'OCCUPIED'),
                            (3,1,'A03','ONE_BEDROOM',1,10000,'VACANT');
    INSERT INTO tenant VALUES (1,'Amina Otieno','+254700000001','en','SALARIED','2025-01-01'),
                              (2,'Juma Kamau','+254700000002','sw','BUSINESS','2025-01-01');
    INSERT INTO lease VALUES (1,1,1,'2025-01-01',NULL,10000,20000,'ACTIVE'),(2,2,2,'2025-01-01',NULL,20000,40000,'ACTIVE');
    INSERT INTO invoice VALUES (1,1,'2026-02-01','2026-02-05',10000,10000,'PAID'),
                               (2,2,'2026-02-01','2026-02-05',20000,0,'UNPAID'),
                               (3,1,'2026-03-01','2026-03-05',10000,10000,'PAID'),
                               (4,2,'2026-03-01','2026-03-05',20000,5000,'PARTIAL');
    INSERT INTO payment VALUES (1,1,1,'2026-02-09',10000,'MPESA','S1'),(2,1,3,'2026-03-04',10000,'MPESA','S2'),
                               (3,2,4,'2026-03-10',5000,'BANK','B1');
    """)
    w.commit()
    w.close()
    c = db.connect_ro(path)
    yield c
    c.close()


def test_occupancy(conn):
    assert metrics.occupancy(conn) == {"occupied": 2, "units": 3, "rate": 2 / 3}


def test_collection_rate(conn):
    r = metrics.collection_rate(conn, MARCH)
    assert (r["expected"], r["collected"], r["invoices"], r["paid_invoices"]) == (30000, 15000, 2, 1)
    assert r["rate"] == 0.5


def test_cash_received_is_by_payment_date(conn):
    assert metrics.cash_received(conn, MARCH) == 15000
    assert metrics.cash_received(conn, dt.date(2026, 2, 1)) == 10000


def test_arrears_and_goal_metric(conn):
    r = metrics.arrears_rate(conn, MARCH)
    assert (r["amount"], r["invoices"], r["tenants"]) == (35000, 2, 1)
    assert r["rate"] == pytest.approx(35000 / 30000)


def test_aging_buckets(conn):
    b = metrics.aging_buckets(conn)
    assert b["8-30"] == {"invoices": 1, "amount": 15000}   # March: due the 5th, 15 days at the 20th
    assert b["31-60"] == {"invoices": 1, "amount": 20000}  # February: 43 days
    assert b["1-7"]["invoices"] == 0 and b["60+"]["invoices"] == 0


def test_days_late_of_a_paid_invoice_uses_the_payment_date(conn):
    assert db.run_select("SELECT days_late FROM v_invoice_state WHERE invoice_id = 1", conn)["rows"] == [[4]]


def test_sql_results_hide_contact_details(conn):
    r = db.run_select("SELECT name, phone, 'call 0712345678 or a@b.co' AS note FROM tenant WHERE id = 1", conn)
    assert r["rows"][0] == ["Amina Otieno", "[hidden]", "call [hidden] or [hidden]"]
