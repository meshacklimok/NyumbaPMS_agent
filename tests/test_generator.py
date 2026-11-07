from nyumba_agent import config, db, metrics
from nyumba_agent.data.generate import generate


def test_same_seed_same_data(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    assert generate(a, 7) == generate(b, 7)


def test_integrity_and_goal_is_meaningful(tmp_path):
    p = tmp_path / "n.db"
    generate(p, 42)
    c = db.connect_ro(p)
    q = lambda s: db.run_select(s, c)["rows"]
    assert q("SELECT COUNT(*) FROM invoice i WHERE amount_paid != (SELECT COALESCE(SUM(amount),0) FROM payment WHERE invoice_id=i.id)") == [[0]]
    assert q("SELECT COUNT(*) FROM payment WHERE paid_on > '%s'" % config.TODAY) == [[0]]
    this_month = config.TODAY.replace(day=1)
    rate = metrics.arrears_rate(c, this_month)["rate"]
    assert 0.10 < rate < 0.30  # above the 10% target, so the goal has work to do
    late = q("SELECT AVG(days_late > %d) FROM v_invoice_state WHERE period_start < '%s'"
             % (config.GRACE_DAYS, this_month))[0][0]
    assert 0.12 < late < 0.35  # enough positives to learn from, not trivial
    assert q("SELECT COUNT(*) FROM unit WHERE status='OCCUPIED'") == q(
        "SELECT COUNT(DISTINCT unit_id) FROM lease WHERE status='ACTIVE'")
    assert q("SELECT COUNT(*) FROM maintenance_request WHERE description LIKE 'Ignore all previous%'") == [[1]]
