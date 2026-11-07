import pytest

from nyumba_agent import config


@pytest.fixture(autouse=True)
def _isolated_agent_db(tmp_path, monkeypatch):
    """No test may touch the real var/agent.db (outbox, feedback, run log)."""
    monkeypatch.setattr(config, "AGENT_DB", tmp_path / "agent_test.db")
