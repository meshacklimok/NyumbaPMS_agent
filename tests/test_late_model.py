"""Late-payment model: no leakage from the future, a sane time split, and the predict_late tool."""

import pandas as pd
import pytest

from nyumba_agent import db
from nyumba_agent.data.generate import generate
from nyumba_agent.ml import late_model as lm
from nyumba_agent.tools.domain_tools import build_registry


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(p, seed=42)
    return p


@pytest.fixture(scope="module")
def frames(path):
    conn = db.connect_ro(path)
    try:
        return lm.load_frames(conn)
    finally:
        conn.close()


def _target(inv, invoice_id):
    row = inv[inv["invoice_id"] == invoice_id].iloc[0]
    t = pd.DataFrame([{"tenant_id": row.tenant_id, "asof": row.period_start, "period_start": row.period_start,
                       "rent": row.total, "list_rent": row.list_rent, "start_date": row.start_date,
                       "employment": row.employment, "property_category": row.property_category}])
    return t


def test_payments_after_the_as_of_date_cannot_change_features(frames):
    inv, pay = frames
    full = lm.add_labels(inv, pay)
    checked = 0
    for invoice_id in full[full["period_start"] >= "2026-01-01"]["invoice_id"].head(150):
        t = _target(full, invoice_id)
        asof = t["asof"].iloc[0]
        trimmed = lm.add_labels(inv, pay[pay["paid_on"] <= asof])  # the world as it was at asof
        a, b = lm.features_as_of(full, t), lm.features_as_of(trimmed, t)
        pd.testing.assert_frame_equal(a, b, check_exact=False)
        checked += 1
    assert checked == 150


def test_first_invoice_has_no_history(frames):
    inv, pay = frames
    full = lm.add_labels(inv, pay)
    first = full.sort_values("period_start").groupby("tenant_id").head(1).iloc[0]
    f = lm.features_as_of(full, _target(full, first.invoice_id)).iloc[0]
    assert f["n_prior"] == 0 and pd.isna(f["late_rate"]) and f["balance"] == 0


def test_time_split_is_ordered_and_labels_are_known(path):
    conn = db.connect_ro(path)
    df = lm.training_frame(conn)
    conn.close()
    train, valid, test = lm.split_by_time(df)
    assert train["period"].max() < valid["period"].min() <= valid["period"].max() < test["period"].min()
    assert set(df["late"]) == {0, 1}


def test_evaluation_reports_both_baselines(path):
    conn = db.connect_ro(path)
    r = lm.evaluate(lm.training_frame(conn))
    conn.close()
    assert set(r["table"]) == {"always on-time", "late if late last month", "logistic regression"}
    assert r["table"]["always on-time"]["recall"] == 0
    assert r["table"]["logistic regression"]["roc_auc"] > 0.5


def test_predict_late_tool(path):
    reg = build_registry(path)
    r = reg.call("predict_late", {"top_n": 5})
    assert "error" not in r and r["month"] == "2026-11" and len(r["top"]) == 5
    probs = [x["probability_late"] for x in r["top"]]
    assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert "phone" not in str(r)
    assert "error" in reg.call("predict_late", {"top_n": 0})
