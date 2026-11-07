import pytest

from nyumba_agent import db
from nyumba_agent.data.generate import generate


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "nyumba.db"
    generate(path, seed=42)
    c = db.connect_ro(path)
    yield c
    c.close()


def test_select_works(conn):
    r = db.run_select("SELECT COUNT(*) AS n FROM unit", conn)
    assert r["rows"] == [[120]] and r["columns"] == ["n"]


@pytest.mark.parametrize("q", [
    "DROP TABLE unit", "DELETE FROM payment", "UPDATE invoice SET total = 0",
    "INSERT INTO property VALUES (9,'x','x','x','x')", "ATTACH DATABASE '/tmp/x.db' AS x",
    "PRAGMA writable_schema = 1", "SELECT 1; DROP TABLE unit", "", "CREATE TABLE t(a)",
])
def test_writes_and_extras_are_refused(conn, q):
    assert "error" in db.run_select(q, conn)


def test_cte_that_writes_is_refused(conn):
    assert "error" in db.run_select("WITH x AS (SELECT 1) DELETE FROM unit", conn)


def test_data_unchanged_after_attacks(conn):
    db.run_select("DROP TABLE unit", conn)
    assert db.run_select("SELECT COUNT(*) FROM unit", conn)["rows"] == [[120]]


def test_row_cap(conn):
    r = db.run_select("SELECT * FROM invoice", conn, limit=10)
    assert r["row_count"] == 10 and r["truncated"]


def test_timeout(conn):
    q = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"
    r = db.run_select(q, conn, timeout=0.3)
    assert "stopped" in r["error"]


def test_bad_column_error_is_readable(conn):
    assert "no such column" in db.run_select("SELECT nope FROM unit", conn)["error"]


def test_file_level_readonly(tmp_path):
    p = tmp_path / "n.db"
    generate(p, seed=1)
    import sqlite3
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        c.execute("DELETE FROM unit")
