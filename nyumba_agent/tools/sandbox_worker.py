"""Runs the model's pandas code. Executed as a separate process by `pandas_tool`; never imported by the agent.

Layers, outermost first (docs/02 guardrails):
  1. the parent kills this process on timeout and gives it an empty environment and an empty working directory;
  2. resource limits (CPU, memory, no file larger than 0 bytes);
  3. `check_code`: an AST allowlist-ish check (imports, builtins, private attributes, file readers/writers);
  4. restricted builtins, and `__import__` that only lets pandas and numpy through;
  5. a Python audit hook that refuses files, sockets and processes even if 3 and 4 are bypassed.
It must stay self-contained (no nyumba_agent imports) because it runs with `python -I`.
"""

import ast
import io
import json
import math
import os
import pickle
import sys

ALLOWED_IMPORTS = {"pandas", "numpy"}
RESULT_ROWS = 200
STDOUT_CHARS = 2000

_BAD_NAMES = {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars", "dir",
              "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit", "type", "super", "object"}
# Attribute names that read or write outside the process, or reach the interpreter.
_BAD_ATTRS = {"format", "format_map", "os", "sys", "io", "subprocess", "shutil", "socket", "ctypes", "builtins",
              "importlib", "pickle", "compat", "testing", "ctypeslib", "f2py", "system", "popen",
              "to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_sql", "to_hdf", "to_feather", "to_stata",
              "to_html", "to_latex", "to_markdown", "to_clipboard", "to_xml", "to_orc", "to_gbq", "to_string",
              "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "tofile",
              "memmap", "fromregex", "eval", "query"}
_ERRORS = ("Exception", "ValueError", "KeyError", "TypeError", "IndexError", "ZeroDivisionError", "AttributeError",
           "ArithmeticError", "LookupError", "StopIteration")
_BUILTINS = ("abs all any bool dict divmod enumerate filter float frozenset int isinstance len list map max min "
             "pow range repr reversed round set slice sorted str sum tuple zip None True False").split()


def check_code(code: str) -> str | None:
    """Return a reason the code is refused, or None. Pure function, so the parent can pre-check cheaply."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error on line {exc.lineno}: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name}' is not allowed (only pandas and numpy)"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed (only pandas and numpy)"
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in _BAD_NAMES:
                return f"name '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return f"attribute '{node.attr}' is not allowed (private attributes are blocked)"
            if node.attr in _BAD_ATTRS or node.attr.startswith("read_"):
                return f"'.{node.attr}' is not allowed (no file, network or string-format access; use f-strings)"
        elif isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            return f"{type(node).__name__} is not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return "strings containing '__' are not allowed"
    return None


def audit_policy(event: str, args: tuple, allowed_read_prefixes: tuple = ()) -> None:
    """Raise PermissionError for events that touch files, network or processes. Pure, so it is unit-testable."""
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return  # an already-open file descriptor
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                  (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            raise PermissionError("writing files is not allowed")
        real = os.path.realpath(os.fsdecode(path))
        if not real.startswith(allowed_read_prefixes):
            raise PermissionError("reading files is not allowed")
        return
    blocked = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
               "os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir", "os.mkdir", "os.chmod", "os.chown",
               "os.truncate", "os.symlink", "os.link", "os.chdir", "os.listdir", "os.scandir", "os.kill", "os.putenv",
               "shutil.", "ctypes.", "sqlite3.", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "glob.",
               "tempfile.", "pickle.", "fcntl.", "mmap.")
    if event.startswith(blocked):
        raise PermissionError(f"'{event}' is not allowed in the sandbox")


def _read_prefixes() -> tuple:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr/share/zoneinfo", "/usr/lib", "/lib"}
    return tuple(os.path.realpath(r) for r in roots if r)


def _limit_resources(cpu_s: int) -> None:
    try:
        import resource
    except ImportError:  # not Unix: the parent timeout and the audit hook remain
        return
    for name, value in (("RLIMIT_CPU", cpu_s), ("RLIMIT_AS", 3 * 1024**3), ("RLIMIT_FSIZE", 0), ("RLIMIT_CORE", 0)):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError):
            pass


def _jsonable(value, np, pd, limit: int):
    """Turn whatever `result` is into JSON: {kind, ...}. Tables are capped; contact details are masked by the parent."""
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        truncated = len(value) > limit
        head = value.head(limit)
        cols = [str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in head.columns]
        if head.index.name or not isinstance(head.index, pd.RangeIndex):
            head = head.reset_index()
            cols = [str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in head.columns]
        rows = [[_scalar(v, np, pd) for v in r] for r in head.itertuples(index=False, name=None)]
        return {"kind": "table", "columns": cols, "rows": rows, "row_count": len(rows),
                "total_rows": int(len(value)), "truncated": truncated}
    return {"kind": "value", "value": _scalar(value, np, pd)}


def _scalar(v, np, pd):
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _scalar(x, np, pd) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)) or isinstance(v, np.ndarray):
        return [_scalar(x, np, pd) for x in list(v)[:1000]]
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return _jsonable(v, np, pd, RESULT_ROWS)
    return str(v)


def main() -> None:
    payload = pickle.loads(sys.stdin.buffer.read())  # trusted: written by our own parent
    code, tables, limit, timeout = payload["code"], payload["tables"], payload["limit"], payload["timeout"]
    out = sys.stdout  # keep the real stdout for the reply; the user's print goes to a buffer
    reason = check_code(code)
    if reason:
        out.write(json.dumps({"error": reason}))
        return
    import numpy as np
    import pandas as pd
    _limit_resources(int(timeout) + 2)

    printed = io.StringIO()

    def safe_print(*args, **kwargs):
        print(*args, **{k: v for k, v in kwargs.items() if k in ("sep", "end")}, file=printed)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import of '{name}' is not allowed (only pandas and numpy)")
        return __import__(name, globals, locals, fromlist, level)

    import builtins
    allowed = {n: getattr(builtins, n) for n in _BUILTINS + list(_ERRORS) if hasattr(builtins, n)}
    allowed.update(print=safe_print, __import__=safe_import)
    namespace = {"__builtins__": allowed, "pd": pd, "np": np, **tables}

    prefixes = _read_prefixes()
    sys.addaudithook(lambda event, args: audit_policy(event, args, prefixes))
    try:
        exec(compile(code, "<analysis>", "exec"), namespace)
    except BaseException as exc:  # includes MemoryError, SystemExit from sandboxed code
        msg = f"{type(exc).__name__}: {exc}"
        out.write(json.dumps({"error": msg[:600], "stdout": printed.getvalue()[:STDOUT_CHARS]}))
        return
    if "result" not in namespace:
        out.write(json.dumps({"error": "the code ran but did not set a variable named `result`",
                              "stdout": printed.getvalue()[:STDOUT_CHARS]}))
        return
    try:
        reply = _jsonable(namespace["result"], np, pd, limit)
        reply["stdout"] = printed.getvalue()[:STDOUT_CHARS]
        out.write(json.dumps(reply))
    except Exception as exc:
        out.write(json.dumps({"error": f"could not convert `result`: {type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
