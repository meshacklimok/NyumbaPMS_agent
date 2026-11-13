"""run_pandas: the model's analysis code in a separate, locked-down process (docs/02, D-007).

The parent loads the tables read-only (phone numbers dropped), pickles them to the worker and reads back JSON.
Nothing the worker returns is unpickled, and the worker gets an empty environment (no API keys) and an empty
working directory. Every failure comes back as {"error": ...} so the model can fix its code and retry.
"""

import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .. import config, db
from . import sandbox_worker

WORKER = Path(sandbox_worker.__file__).resolve()
_OBJECTS = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"


def load_tables(db_path=None) -> dict[str, pd.DataFrame]:
    conn = db.connect_ro(db_path)
    try:
        names = [r[0] for r in conn.execute(_OBJECTS)]
        return {n: pd.read_sql_query(f"SELECT * FROM {n}", conn).drop(columns=config.HIDDEN_COLUMNS.get(n, []))
                for n in names}
    finally:
        conn.close()


def _mask(obj):
    if isinstance(obj, str):
        return db.mask(obj)
    if isinstance(obj, list):
        return [_mask(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _mask(v) for k, v in obj.items()}
    return obj


def run_pandas(code: str, db_path=None, timeout: float = config.PANDAS_TIMEOUT_S,
               limit: int = config.SQL_ROW_LIMIT, tables: dict | None = None) -> dict:
    """Run `code` with the tables as DataFrames named after the tables, `pd` and `np`. It must set `result`."""
    if not isinstance(code, str) or not code.strip():
        return {"error": "empty code"}
    if len(code) > config.PANDAS_MAX_CODE_CHARS:
        return {"error": f"code is longer than {config.PANDAS_MAX_CODE_CHARS} characters; split it up"}
    reason = sandbox_worker.check_code(code)  # fail fast without starting a process
    if reason:
        return {"error": reason}
    try:
        tables = tables if tables is not None else load_tables(db_path)
    except Exception as exc:
        return {"error": f"could not load the data: {exc}"}
    payload = pickle.dumps({"code": code, "tables": tables, "limit": limit, "timeout": timeout})
    with tempfile.TemporaryDirectory() as cwd:
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
        proc = subprocess.Popen([sys.executable, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd, env=env, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.communicate()
            return {"error": f"code took longer than {timeout:g}s and was stopped (infinite loop or too heavy?)"}
        finally:
            _kill(proc)
    try:
        reply = json.loads(stdout.decode("utf-8", "replace"))
    except ValueError:
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        return {"error": f"the sandbox process failed (exit {proc.returncode}): {tail[0][:300]}"}
    return _mask(reply)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
