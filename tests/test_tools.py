"""Chart validation, registry behaviour and the default tools. Every failure must be data, never an exception."""

import json

import pytest

from nyumba_agent.data.generate import generate
from nyumba_agent.tools.domain_tools import build_registry
from nyumba_agent.tools.registry import Registry, Tool, model_view


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(path, seed=42)
    return build_registry(path)


BAR_SQL = "SELECT p.name AS property, COUNT(*) AS units FROM unit u JOIN property p ON p.id = u.property_id GROUP BY p.name"


def test_chart_renders_from_sql(reg):
    r = reg.call("make_chart", {"type": "bar", "sql": BAR_SQL, "x": "property", "y": "units", "title": "Units"})
    assert r["ok"] and r["rows_plotted"] == 3
    fig = json.loads(r["_figure"])
    assert fig["layout"]["title"]["text"] == "Units" and fig["data"][0]["type"] == "bar"


def test_chart_can_plot_the_last_pandas_table(reg):
    reg.call("run_pandas", {"code": "result = unit.groupby('status').size().rename('n')"})
    r = reg.call("make_chart", {"type": "pie", "use_last_result": True, "x": "status", "y": "n", "title": "Status"})
    assert r["ok"], r


@pytest.mark.parametrize("spec,fragment", [
    ({"type": "donut", "sql": BAR_SQL, "x": "property", "y": "units", "title": "t"}, "is not one of"),
    ({"type": "bar", "sql": BAR_SQL, "x": "propertee", "y": "units", "title": "t"}, "x column 'propertee'"),
    ({"type": "bar", "sql": BAR_SQL, "x": "property", "y": "property", "title": "t"}, "numeric"),
    ({"type": "bar", "sql": BAR_SQL, "x": "property", "title": "t"}, "needs a y column"),
    ({"type": "bar", "x": "a", "y": "b", "title": "t"}, "sql or use_last_result"),
    ({"type": "bar", "sql": "DROP TABLE unit", "x": "a", "y": "b", "title": "t"}, "query failed"),
    ({"type": "bar", "sql": "SELECT 1 AS a, 2 AS b WHERE 0", "x": "a", "y": "b", "title": "t"}, "no rows"),
    ({"type": "bar", "sql": BAR_SQL, "x": "property", "y": "units", "title": "t" * 200}, "title"),
    ({"type": "bar", "sql": BAR_SQL, "x": "property", "y": "units", "title": "t", "script": "x"}, "invalid arguments"),
])
def test_bad_chart_specs_return_errors(reg, spec, fragment):
    r = reg.call("make_chart", spec)
    assert fragment in r["error"]


def test_chart_error_lists_real_columns_so_the_model_can_retry(reg):
    r = reg.call("make_chart", {"type": "bar", "sql": BAR_SQL, "x": "name", "y": "units", "title": "t"})
    assert "property, units" in r["error"]


def test_model_view_hides_figures_and_truncates():
    assert "_figure" not in model_view({"ok": True, "_figure": "x" * 100000})
    assert "cut" in model_view({"rows": ["x" * 100] * 200}, max_chars=500)


def test_registry_errors_are_data():
    boom = Tool("boom", "d", {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
                lambda a: 1 / 0)
    r = Registry([boom])
    assert "unknown tool" in r.call("nope", {})["error"]
    assert "invalid arguments" in r.call("boom", {})["error"]
    assert "invalid arguments for n" in r.call("boom", {"n": "x"})["error"]
    assert "ZeroDivisionError" in r.call("boom", {"n": 1})["error"]
    assert "JSON object" in r.call("boom", "not a dict")["error"]
    with pytest.raises(ValueError):
        r.register(boom)


def test_specs_match_anthropic_shape(reg):
    specs = reg.specs()
    assert {s["name"] for s in specs} == {"run_sql", "run_pandas", "make_chart", "get_metric", "predict_late", "draft_reminder"}
    assert all(set(s) == {"name", "description", "input_schema"} for s in specs)


def test_sql_tool_wraps_the_guard(reg):
    assert reg.call("run_sql", {"query": "SELECT COUNT(*) AS n FROM unit"})["rows"] == [[120]]
    assert "error" in reg.call("run_sql", {"query": "DELETE FROM unit"})
    assert "no such column" in reg.call("run_sql", {"query": "SELECT rent_amount FROM unit"})["error"]


def test_metric_tool(reg):
    occ = reg.call("get_metric", {"name": "occupancy"})
    assert occ["value"]["units"] == 120
    rate = reg.call("get_metric", {"name": "arrears_rate", "month": "2026-10"})
    assert 0 < rate["value"]["rate"] < 1
    assert "month must look like" in reg.call("get_metric", {"name": "arrears_rate", "month": "Oct"})["error"]
    assert "portfolio-wide" in reg.call("get_metric", {"name": "occupancy", "property_id": 1})["error"]
    assert "invalid arguments" in reg.call("get_metric", {"name": "profit"})["error"]
