"""make_chart: the model gives a declarative spec, never plotting code (D-007). We validate it against the
data and render it with Plotly. The figure JSON travels in `_figure`; the loop keeps it for the UI and hides
it from the model (keys starting with `_` are not shown to the model).
"""

import pandas as pd
import plotly.express as px

from .. import config, db

CHART_TYPES = ("bar", "line", "scatter", "pie", "histogram", "box")
# Categorical order from the validated dataviz palette (blue, orange, aqua, yellow, magenta, green, violet, red).
COLORWAY = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(CHART_TYPES)},
        "sql": {"type": "string", "description": "A SELECT that returns the data to plot. Use this or use_last_result."},
        "use_last_result": {"type": "boolean",
                            "description": "Plot the table returned by the previous run_pandas call instead of running SQL."},
        "x": {"type": "string", "description": "Column for the x axis (pie: slice names; histogram: the values)."},
        "y": {"type": "string", "description": "Numeric column for the y axis (pie: slice sizes). Not used by histogram."},
        "color": {"type": "string", "description": "Optional column that splits the data into coloured series."},
        "title": {"type": "string", "description": "Chart title, under 120 characters."},
    },
    "required": ["type", "x", "title"],
    "additionalProperties": False,
}


def make_chart(args: dict, db_path=None, last_table: pd.DataFrame | None = None) -> dict:
    kind, x, y, color, title = (args.get(k) for k in ("type", "x", "y", "color", "title"))
    if kind not in CHART_TYPES:
        return {"error": f"type must be one of {', '.join(CHART_TYPES)}"}
    if not title or len(title) > 120:
        return {"error": "title is required and must be under 120 characters"}
    if kind not in ("histogram",) and not y:
        return {"error": f"a {kind} chart needs a y column"}

    if args.get("use_last_result"):
        if last_table is None:
            return {"error": "there is no previous table result; run run_pandas first or give a sql query"}
        df = last_table
    elif args.get("sql"):
        conn = db.connect_ro(db_path)
        try:
            r = db.run_select(args["sql"], conn, limit=config.CHART_MAX_ROWS)
        finally:
            conn.close()
        if "error" in r:
            return {"error": f"the chart query failed: {r['error']}"}
        df = pd.DataFrame(r["rows"], columns=r["columns"])
    else:
        return {"error": "give either sql or use_last_result"}

    if df.empty:
        return {"error": "the data has no rows, so there is nothing to plot"}
    if len(df) > config.CHART_MAX_ROWS:
        return {"error": f"the data has {len(df)} rows; aggregate to at most {config.CHART_MAX_ROWS} first"}
    for label, col in (("x", x), ("y", y), ("color", color)):
        if col and col not in df.columns:
            return {"error": f"{label} column '{col}' is not in the data; columns are: {', '.join(map(str, df.columns))}"}
    if y and kind != "histogram" and not pd.api.types.is_numeric_dtype(df[y]):
        return {"error": f"y column '{y}' must be numeric, but it is {df[y].dtype}"}
    if kind == "pie" and color:
        return {"error": "a pie chart has no color column; the slices are the x values"}

    common = dict(title=title, color_discrete_sequence=COLORWAY)
    try:
        if kind == "bar":
            fig = px.bar(df, x=x, y=y, color=color, barmode="group", **common)
        elif kind == "line":
            fig = px.line(df, x=x, y=y, color=color, markers=True, **common)
        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, **common)
        elif kind == "pie":
            fig = px.pie(df, names=x, values=y, **common)
        elif kind == "histogram":
            fig = px.histogram(df, x=x, color=color, **common)
        else:
            fig = px.box(df, x=color or x, y=y, **common) if color else px.box(df, x=x, y=y, **common)
    except Exception as exc:  # Plotly raises ValueError for combinations we did not foresee
        return {"error": f"could not draw that chart: {type(exc).__name__}: {exc}"}

    fig.update_layout(template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=MUTED), title=dict(font=dict(color=INK, size=16)), margin=dict(t=60, r=20),
                      legend_title_text="")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return {"ok": True, "title": title, "type": kind, "rows_plotted": len(df), "_figure": fig.to_json()}
