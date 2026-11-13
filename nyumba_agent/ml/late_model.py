"""Late-payment classifier (docs/04, "Learning"). Unit of prediction: one invoice. Label: paid in full after
due_date + GRACE_DAYS, or still unpaid.

Leakage rule: every feature for an invoice is computed from the tenant's *earlier* invoices, using only payments
made on or before the as-of date (the invoice's period start, which is before its due date). `features_as_of`
is the one place that rule is enforced, and tests/test_late_model.py checks it by deleting later payments.
"""

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config, db

NUMERIC = ["n_prior", "late_rate", "avg_days_late", "last_late", "late_last3", "balance",
           "rent", "rent_ratio", "tenure_months", "is_jan", "is_dec_feb"]
CATEGORICAL = ["employment", "property_category"]
FEATURES = NUMERIC + CATEGORICAL
TEST_MONTHS = 3  # the newest months are the test set
VALID_MONTHS = 3  # the months just before them pick the decision threshold


def load_frames(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_sql_query(
        "SELECT i.id AS invoice_id, l.tenant_id, l.id AS lease_id, i.period_start, i.due_date, i.total, "
        "u.list_rent, l.start_date, t.employment, p.category AS property_category, p.id AS property_id "
        "FROM invoice i JOIN lease l ON l.id = i.lease_id JOIN unit u ON u.id = l.unit_id "
        "JOIN property p ON p.id = u.property_id JOIN tenant t ON t.id = l.tenant_id", conn)
    pay = pd.read_sql_query("SELECT invoice_id, paid_on, amount FROM payment WHERE invoice_id IS NOT NULL", conn)
    for col in ("period_start", "due_date", "start_date"):
        inv[col] = pd.to_datetime(inv[col])
    pay["paid_on"] = pd.to_datetime(pay["paid_on"])
    return inv, pay


def _paid_in_full_dates(inv: pd.DataFrame, pay: pd.DataFrame) -> pd.Series:
    """Date each invoice's running payments first reached its total (NaT: never, as far as the data goes)."""
    p = pay.sort_values(["invoice_id", "paid_on"]).merge(inv[["invoice_id", "total"]], on="invoice_id")
    p["cum"] = p.groupby("invoice_id")["amount"].cumsum()
    full = p[p["cum"] >= p["total"]].groupby("invoice_id")["paid_on"].min()
    return inv["invoice_id"].map(full)


def add_labels(inv: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """paid_full_on and `late`. late is NaN while the grace period has not ended yet (outcome unknown)."""
    inv = inv.copy()
    inv["paid_full_on"] = _paid_in_full_dates(inv, pay)
    cutoff = inv["due_date"] + pd.Timedelta(days=config.GRACE_DAYS)
    late = inv["paid_full_on"].isna() | (inv["paid_full_on"] > cutoff)
    inv["late"] = late.astype(float).where(cutoff <= pd.Timestamp(config.TODAY))
    return inv


def _history_row(prior: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Summaries of one tenant's earlier invoices, seeing only what happened by `asof`."""
    paid = prior["paid_full_on"].where(prior["paid_full_on"] <= asof)  # later payments are invisible
    cutoff = prior["due_date"] + pd.Timedelta(days=config.GRACE_DAYS)
    decided = cutoff <= asof  # grace period over, so late-or-not is known at asof
    late = (paid.isna() | (paid > cutoff))[decided]
    days = (paid.fillna(asof) - prior["due_date"]).dt.days.clip(lower=0)[decided]
    unpaid_total = prior["total"].where(paid.isna(), 0).sum()
    n = int(decided.sum())
    return {"n_prior": n,
            "late_rate": float(late.mean()) if n else np.nan,
            "avg_days_late": float(days.mean()) if n else np.nan,
            "last_late": float(late.iloc[-1]) if n else np.nan,  # prior is sorted by period
            "late_last3": float(late.iloc[-3:].sum()) if n else np.nan,
            "balance": float(unpaid_total)}


def features_as_of(inv: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """One feature row per target. `inv` must come from add_labels; `targets` has tenant_id, asof, period_start,
    rent, list_rent, start_date, employment, property_category (and invoice_id when training)."""
    inv = inv.sort_values("period_start")
    by_tenant = {tid: g for tid, g in inv.groupby("tenant_id")}
    rows = []
    for t in targets.itertuples():
        g = by_tenant.get(t.tenant_id)
        prior = g[g["period_start"] < t.period_start] if g is not None else inv.iloc[0:0]
        h = _history_row(prior, t.asof)
        month = t.period_start.month
        rows.append({**h,
                     
                     "rent": t.rent, "rent_ratio": t.rent / t.list_rent,
                     "tenure_months": (t.period_start.year - t.start_date.year) * 12 + t.period_start.month - t.start_date.month,
                     "is_jan": int(month == 1), "is_dec_feb": int(month in (12, 2)),
                     "employment": t.employment, "property_category": t.property_category})
    return pd.DataFrame(rows, index=targets.index)[FEATURES]


def training_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every invoice with a known outcome: its features as of period start, its label and its month."""
    inv, pay = load_frames(conn)
    inv = add_labels(inv, pay)
    known = inv[inv["late"].notna()].copy()
    known["asof"] = known["period_start"]
    known["rent"] = known["total"]
    X = features_as_of(inv, known)
    out = X.assign(late=known["late"].astype(int), period=known["period_start"], invoice_id=known["invoice_id"])
    return out.sort_values(["period", "invoice_id"]).reset_index(drop=True)


def make_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def pick_threshold(y, p) -> float:
    grid = np.arange(0.2, 0.81, 0.02)
    return float(grid[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid]))])


def split_by_time(df: pd.DataFrame):
    months = sorted(df["period"].unique())
    test_from, valid_from = months[-TEST_MONTHS], months[-TEST_MONTHS - VALID_MONTHS]
    return (df[df["period"] < valid_from], df[(df["period"] >= valid_from) & (df["period"] < test_from)],
            df[df["period"] >= test_from])


def _scores(y, pred, prob=None) -> dict:
    out = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0), "accuracy": float((np.asarray(y) == np.asarray(pred)).mean())}
    if prob is not None:
        out["roc_auc"], out["pr_auc"] = roc_auc_score(y, prob), average_precision_score(y, prob)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    """Train on old months, pick the threshold on the next three, refit on both, score on the newest three."""
    train, valid, test = split_by_time(df)
    probe = make_model().fit(train[FEATURES], train["late"])
    threshold = pick_threshold(valid["late"], probe.predict_proba(valid[FEATURES])[:, 1])
    model = make_model().fit(pd.concat([train, valid])[FEATURES], pd.concat([train, valid])["late"])
    prob = model.predict_proba(test[FEATURES])[:, 1]
    y = test["late"].to_numpy()
    last_late = test["last_late"].fillna(0).to_numpy()  # no history counts as "was not late"
    table = {"always on-time": _scores(y, np.zeros_like(y)),
             "late if late last month": _scores(y, last_late.astype(int)),
             "logistic regression": _scores(y, (prob >= threshold).astype(int), prob)}
    coefs = _coefficients(model)
    return {"table": table, "threshold": threshold, "coefficients": coefs,
            "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            "late_rate": {"train": train["late"].mean(), "test": float(y.mean())},
            "periods": {"train_to": str(train["period"].max().date()), "test_from": str(test["period"].min().date()),
                        "test_to": str(test["period"].max().date())},
            "beats_baselines": bool(all(table["logistic regression"]["f1"] > table[b]["f1"]
                                        for b in ("always on-time", "late if late last month")))}


def _coefficients(model: Pipeline) -> list[tuple[str, float]]:
    names = list(model.named_steps["pre"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    return sorted(zip(names, map(float, coef)), key=lambda kv: -abs(kv[1]))


def fit_final(conn: sqlite3.Connection) -> dict:
    """Everything the tool needs: the model fitted on all known invoices and the validated threshold."""
    df = training_frame(conn)
    train, valid, _ = split_by_time(df)
    probe = make_model().fit(train[FEATURES], train["late"])
    threshold = pick_threshold(valid["late"], probe.predict_proba(valid[FEATURES])[:, 1])
    return {"model": make_model().fit(df[FEATURES], df["late"]), "threshold": threshold, "trained_on": len(df)}


def predict_next_month(conn: sqlite3.Connection, bundle: dict, property_id: int | None = None) -> pd.DataFrame:
    """Score every active lease for next month's invoice, using only what is known today."""
    inv, pay = load_frames(conn)
    inv = add_labels(inv, pay)
    today = pd.Timestamp(config.TODAY)
    nxt = (today.replace(day=1) + pd.offsets.MonthBegin(1))
    q = ("SELECT l.id AS lease_id, l.tenant_id, l.rent, l.start_date, u.code AS unit, u.list_rent, u.property_id, "
         "t.name AS tenant, t.employment, p.category AS property_category FROM lease l JOIN unit u ON u.id = l.unit_id "
         "JOIN tenant t ON t.id = l.tenant_id JOIN property p ON p.id = u.property_id WHERE l.status = 'ACTIVE'")
    cur = pd.read_sql_query(q, conn)
    if property_id is not None:
        cur = cur[cur["property_id"] == property_id]
    cur["start_date"] = pd.to_datetime(cur["start_date"])
    cur["period_start"], cur["asof"] = nxt, today
    X = features_as_of(inv, cur)
    cur["probability"] = bundle["model"].predict_proba(X[FEATURES])[:, 1]
    cur["likely_late"] = cur["probability"] >= bundle["threshold"]
    cur["past_late_rate"], cur["unpaid_balance"], cur["days_late_avg"] = X["late_rate"], X["balance"], X["avg_days_late"]
    return cur.sort_values("probability", ascending=False)
