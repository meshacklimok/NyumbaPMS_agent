"""The Ask tab, driven headless against the scripted demo model: no key, no network."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nyumba_agent import config
from nyumba_agent.data.generate import generate
from nyumba_agent.ui import session


@pytest.fixture
def app(tmp_path, monkeypatch):
    generate(tmp_path / "nyumba.db", seed=42)
    monkeypatch.setattr(config, "NYUMBA_DB", tmp_path / "nyumba.db")
    monkeypatch.setattr(config, "AGENT_DB", tmp_path / "agent.db")
    monkeypatch.setenv("NYUMBA_LLM", "fake")
    st.cache_resource.clear()  # the page caches its registry, which is bound to one database
    return AppTest.from_file(str(config.ROOT / "nyumba_agent/ui/app.py"), default_timeout=30)


def test_backend_is_one_setting(monkeypatch):
    monkeypatch.delenv("NYUMBA_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert session.backend() == "fake"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert session.backend() == "openai"
    monkeypatch.setenv("NYUMBA_LLM", "fake")
    assert session.backend() == "fake"


def test_demo_question_shows_answer_error_retry_and_chart(app):
    app.run()
    assert not app.exception
    app.chat_input[0].set_value(session.DEMO_QUESTION).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "worst collection rate" in text
    assert "1 retried after an error" in app.expander[0].label
    assert len(app.get("plotly_chart")) == 1
    assert any("error" in m.value for m in app.markdown)  # the failed first step is visible in the trace


def test_unknown_question_in_demo_mode_is_honest(app):
    app.run()
    app.chat_input[0].set_value("How many tenants?").run()
    assert "demo mode" in " ".join(m.value for m in app.markdown)
    assert not app.get("plotly_chart")


def test_outbox_tab_draft_approve_and_send(app):
    app.run()
    assert not app.exception
    app.selectbox[0].select_index(0).run()
    [b for b in app.button if b.label == "Draft reminder"][0].click().run()
    assert not app.exception
    assert any(b.label == "Reject" for b in app.button)
    assert not any(b.label.startswith("Send") for b in app.button)  # nothing sendable before approval
    [b for b in app.button if b.label == "Approve"][0].click().run()
    app.radio[0].set_value("Ready to send").run()
    [b for b in app.button if b.label.startswith("Send")][0].click().run()
    assert any("SIMULATED" in i.value for i in app.info)
