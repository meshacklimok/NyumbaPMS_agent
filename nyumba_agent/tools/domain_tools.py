"""The default registry: run_sql, run_pandas, make_chart, get_metric and predict_late, bound to one database."""

import datetime as dt

import pandas as pd

from .. import config, db, metrics, outbox
from ..ml import late_model
from .chart_tool import SCHEMA as CHART_SCHEMA, make_chart
from .pandas_tool import run_pandas
from .registry import Registry, Tool
from .reminder_tool import make_draft_reminder

METRICS = {
    "occupancy": (metrics.occupancy, False),
    "expected_rent": (metrics.expected_rent, True),
    "collection_rate": (metrics.collection_rate, True),
    "cash_received": (metrics.cash_received, False),
    "arrears": (metrics.arrears, False),
    "arrears_rate": (metrics.arrears_rate, True),
    "aging_buckets": (metrics.aging_buckets, False),
}
_NEEDS_MONTH = {"expected_rent", "collection_rate", "cash_received", "arrears_rate"}
_NO_PROPERTY = {"occupancy", "cash_received"}


def build_registry(db_path=None, agent_db_path=None) -> Registry:
    """Tools share one piece of state: the last table run_pandas returned, so make_chart can plot it."""
    last: dict = {"table": None}
    model: dict = {}  # the late-payment model, trained on first use

    def predict_late(args):
        conn = db.connect_ro(db_path)
        try:
            if "bundle" not in model:
                model["bundle"] = late_model.fit_final(conn)
            bundle = model["bundle"]
            scored = late_model.predict_next_month(conn, bundle, args.get("property_id"))
        finally:
            conn.close()
        # Learning from feedback: tenants whose reminders the landlord keeps rejecting need a higher bar.
        fb = outbox.open_outbox(db.connect_rw(agent_db_path))
        try:
            bars = outbox.nudge_thresholds(fb, scored["tenant_id"].unique(), bundle["threshold"])
        finally:
            fb.close()
        scored["bar"] = scored["tenant_id"].map(bars)
        scored["likely_late"] = scored["probability"] >= scored["bar"]
        top = scored.head(args.get("top_n", 10))
        rows = [{"tenant": r.tenant, "unit": r.unit, "probability_late": round(r.probability, 2),
                 "likely_late": bool(r.likely_late),
                 **({"raised_bar": round(r.bar, 2)} if r.bar > bundle["threshold"] else {}),
                 "past_late_rate": None if r.past_late_rate != r.past_late_rate else round(r.past_late_rate, 2),
                 "unpaid_balance": int(r.unpaid_balance)} for r in top.itertuples()]
        return {"month": str((scored["period_start"].iloc[0]).date())[:7] if len(scored) else None,
                "leases_scored": len(scored), "flagged_likely_late": int(scored["likely_late"].sum()),
                "threshold": round(bundle["threshold"], 2), "top": rows,
                "note": "Estimates from a model trained on synthetic data; precision is about 0.4, so treat them as "
                        "a watch list, not a verdict."}

    def sql(args):
        conn = db.connect_ro(db_path)
        try:
            return db.run_select(args["query"], conn)
        finally:
            conn.close()

    def pandas_(args):
        result = run_pandas(args["code"], db_path)
        last["table"] = (pd.DataFrame(result["rows"], columns=result["columns"])
                         if result.get("kind") == "table" else None)
        return result

    def chart(args):
        return make_chart(args, db_path, last["table"])

    def metric(args):
        name = args["name"]
        fn = METRICS[name][0]
        kwargs = {}
        if name in _NEEDS_MONTH:
            try:
                kwargs["month"] = dt.date.fromisoformat(args.get("month", config.TODAY.strftime("%Y-%m")) + "-01")
            except ValueError:
                return {"error": "month must look like 2026-09"}
        if args.get("property_id") is not None:
            if name in _NO_PROPERTY:
                return {"error": f"{name} is portfolio-wide and does not take property_id"}
            kwargs["property_id"] = args["property_id"]
        conn = db.connect_ro(db_path)
        try:
            return {"metric": name, **{k: str(v) if k == "month" else v for k, v in kwargs.items()},
                    "value": fn(conn, **kwargs)}
        finally:
            conn.close()

    return Registry([
        make_draft_reminder(db_path, agent_db_path),
        Tool("predict_late",
             "Learned model: the chance that each active tenant pays NEXT month's rent late (after the "
             f"{config.GRACE_DAYS}-day grace), from their payment history up to today. Returns the highest-risk tenants "
             "first. Use it for forward-looking questions ('who is likely to be late next month?'); use get_metric "
             "or run_sql for what has already happened.",
             {"type": "object", "properties": {
                 "property_id": {"type": "integer"},
                 "top_n": {"type": "integer", "minimum": 1, "maximum": 50}},
              "additionalProperties": False}, predict_late),
        Tool("run_sql",
             "Run ONE read-only SQLite SELECT (or WITH ... SELECT) on the property database and get up to "
             f"{config.SQL_ROW_LIMIT} rows. Use it for lookups, joins and aggregates. Money is integer KES, dates are "
             "ISO text. On an error, read the message, fix the query and call again.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
              "additionalProperties": False}, sql),
        Tool("run_pandas",
             "Run Python in a locked-down sandbox for analysis SQL cannot do easily (pivots, rolling windows, "
             "statistics). Every table and view is already a DataFrame named after it (unit, invoice, v_arrears, ...), "
             "plus `pd` and `np`. Only pandas and numpy can be imported; no files, no network. Assign the answer to a "
             "variable named `result` (DataFrame, Series or value). Use print() for debugging.",
             {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"],
              "additionalProperties": False}, pandas_),
        Tool("make_chart",
             "Draw a chart from a spec, not code. Supply the data with a SQL query (or use_last_result to plot the "
             "table from the last run_pandas call), then name the columns for x, y and optionally color. "
             "The chart is shown to the user automatically; you only get a confirmation or an error.",
             CHART_SCHEMA, chart),
        Tool("get_metric",
             "Official business metrics, defined once. Prefer this over writing SQL for these: "
             f"{', '.join(METRICS)}. Month is YYYY-MM (default: the current month). arrears_rate is total arrears "
             f"divided by the month's expected rent; the goal is below {config.ARREARS_TARGET:.0%}.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "enum": list(METRICS)},
                 "month": {"type": "string", "description": "YYYY-MM"},
                 "property_id": {"type": "integer"}},
              "required": ["name"], "additionalProperties": False}, metric),
    ])
