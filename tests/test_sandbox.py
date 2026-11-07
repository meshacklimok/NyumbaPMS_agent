"""The pandas sandbox must run real analysis and refuse everything else (docs/02 guardrails)."""

import os
import time

import pandas as pd
import pytest

from nyumba_agent.data.generate import generate
from nyumba_agent.tools import sandbox_worker
from nyumba_agent.tools.pandas_tool import load_tables, run_pandas


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(path, seed=42)
    return load_tables(path)


def run(code, tables, **kw):
    return run_pandas(code, tables=tables, **kw)


def test_real_analysis_returns_a_table(tables):
    r = run("m = invoice.merge(lease, left_on='lease_id', right_on='id')\n"
            "result = m.groupby('unit_id')['total'].sum().sort_values(ascending=False).head(3)", tables)
    assert r["kind"] == "table" and r["row_count"] == 3 and r["columns"][0] == "unit_id"


def test_scalar_result_and_print(tables):
    r = run("print('hello')\nresult = int(unit.shape[0])", tables)
    assert r == {"kind": "value", "value": 120, "stdout": "hello\n"}


def test_numpy_and_allowed_imports(tables):
    r = run("import numpy as np\nimport pandas as pd\nresult = float(np.median(unit['list_rent']))", tables)
    assert r["kind"] == "value" and r["value"] > 0


def test_tables_are_preloaded_and_phone_is_not_exposed(tables):
    assert "phone" not in tables["tenant"].columns
    assert "v_arrears" in tables
    r = run("result = list(tenant.columns)", tables)
    assert "phone" not in r["value"]


def test_row_cap(tables):
    r = run("result = payment", tables, limit=10)
    assert r["row_count"] == 10 and r["truncated"] and r["total_rows"] > 10


def test_missing_result_is_reported(tables):
    assert "result" in run("x = 1", tables)["error"]


def test_wrong_column_comes_back_as_error_data(tables):
    r = run("result = unit['rent_amount'].sum()", tables)
    assert "KeyError" in r["error"] and "rent_amount" in r["error"]


@pytest.mark.parametrize("code", [
    "import os\nresult = os.listdir('/')",
    "import subprocess\nresult = 1",
    "from os import path\nresult = 1",
    "__import__('os').system('id')",
    "result = open('/etc/passwd').read()",
    "result = open('out.txt', 'w').write('x')",
    "unit.to_csv('out.csv')\nresult = 1",
    "result = pd.read_csv('/etc/passwd')",
    "result = pd.read_csv('http://example.com/x.csv')",
    "np.save('x.npy', np.arange(3))\nresult = 1",
    "result = pd.io.common.os.system('id')",
    "result = ().__class__.__bases__[0].__subclasses__()",
    "result = '{0.__class__}'.format(1)",
    "result = getattr(pd, 'o' + 's')",
    "exec('result = 1')",
    "result = eval('1')",
    "class A: pass\nresult = 1",
    "def f(:\n    pass",
])
def test_dangerous_code_is_refused(code, tables):
    r = run(code, tables)
    assert "error" in r and "kind" not in r


def test_infinite_loop_is_killed(tables):
    start = time.monotonic()
    r = run("while True:\n    pass\nresult = 1", tables, timeout=1.5)
    assert "longer than" in r["error"]
    assert time.monotonic() - start < 8


def test_memory_bomb_is_an_error_not_a_crash(tables):
    r = run("result = np.zeros(10**11).sum()", tables)
    assert "error" in r


def test_write_attempt_leaves_no_file(tables, tmp_path):
    target = tmp_path / "nope.txt"
    r = run(f"result = open({str(target)!r}, 'w')", tables)
    assert "error" in r and not target.exists()


def test_mutating_a_table_does_not_leak_between_runs(tables):
    run("unit.drop(unit.index, inplace=True)\nresult = 1", tables)
    assert run("result = len(unit)", tables)["value"] == 120


def test_environment_secrets_are_not_visible(tables, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    # os cannot be imported at all, and the worker process starts with an empty environment
    assert "error" in run("import os\nresult = os.environ.get('ANTHROPIC_API_KEY')", tables)


# The audit hook is the backstop if the AST check is ever bypassed, so test its policy directly.
@pytest.mark.parametrize("event,args", [
    ("open", ("/etc/passwd", "r", 0)), ("open", ("/tmp/x", "w", 0)), ("open", ("/tmp/x", None, os.O_WRONLY | os.O_CREAT)),
    ("socket.connect", ()), ("subprocess.Popen", ()), ("os.system", ()), ("os.remove", ()), ("ctypes.dlopen", ()),
])
def test_audit_policy_blocks(event, args):
    with pytest.raises(PermissionError):
        sandbox_worker.audit_policy(event, args, allowed_read_prefixes=("/usr/lib",))


def test_audit_policy_allows_reading_python_files():
    sandbox_worker.audit_policy("open", ("/usr/lib/python3/x.py", "rb", 0), allowed_read_prefixes=("/usr/lib",))
    sandbox_worker.audit_policy("exec", ())


def test_audit_hook_stops_code_that_gets_past_the_ast_check(tables, tmp_path):
    """Switch the AST check off and run the worker for real: the audit hook alone must still refuse."""
    import json
    import pickle
    import subprocess
    import sys

    target = tmp_path / "escaped.txt"
    code = f"result = pd.io.common.os.system('id') if False else pd.io.common.os.open({str(target)!r}, 65)"
    shim = ("import importlib.util, sys, pickle\n"
            "spec = importlib.util.spec_from_file_location('w', sys.argv[1])\n"
            "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
            "w.check_code = lambda code: None\n"
            "w.main()\n")
    payload = pickle.dumps({"code": code, "tables": {}, "limit": 10, "timeout": 5})
    p = subprocess.run([sys.executable, "-I", "-c", shim, sandbox_worker.__file__], input=payload,
                       capture_output=True, timeout=30)
    reply = json.loads(p.stdout)
    assert "PermissionError" in reply["error"] and not target.exists()
