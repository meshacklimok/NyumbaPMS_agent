"""Streamlit app. Ask tab: question box, answer, trace of tool steps (errors and retries included) and charts.
Outbox tab: drafted reminders the landlord approves, edits or rejects; sending is simulated.
Run: streamlit run nyumba_agent/ui/app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `streamlit run` finds the package

import plotly.io as pio
import streamlit as st

from nyumba_agent import config, db, outbox
from nyumba_agent.agent.run_log import run_logged
from nyumba_agent.tools.registry import model_view
from nyumba_agent.tools.reminder_tool import invoice_facts
from nyumba_agent.ui import session

session.load_env()
backend = session.backend()

st.set_page_config(page_title="NyumbaAgent", page_icon="🏠", layout="wide")
st.title("NyumbaAgent")
st.caption("Property analyst for a Kenyan landlord. All data is synthetic. The agent can only read data.")


@st.cache_resource
def environment():
    return session.build_env()


def show_trace(run) -> None:
    label = f"Trace: {len(run.steps)} tool step(s), {run.retries} retried after an error, {run.usage['model_calls']} model call(s)"
    with st.expander(label, expanded=any(s.is_error for s in run.steps)):
        for i, s in enumerate(run.steps, 1):
            st.markdown(f"**{i}. `{s.tool}`** " + (":red[error]" if s.is_error else ":green[ok]"))
            st.code(json.dumps(s.input, indent=2), language="json")
            out = json.loads(model_view(s.result)) if not s.is_error else s.result
            st.code(json.dumps(out, indent=2, default=str)[:1500], language="json")
        if not run.steps:
            st.write("No tools were used.")


def show_run(run, key: str) -> None:
    with st.chat_message("user"):
        st.write(run.question)
    with st.chat_message("assistant"):
        if run.stop_reason != "answered":
            st.warning(f"Run stopped: {run.stop_reason}")
        st.markdown(run.answer)
        for n, art in enumerate(run.artifacts):
            st.plotly_chart(pio.from_json(art["figure"]), width="stretch", key=f"{key}-chart-{n}")
        show_trace(run)


with st.sidebar:
    st.subheader("Model")
    if backend == "fake":
        st.info("Demo mode: scripted model, no key needed. Set `NYUMBA_LLM=openai` and `OPENAI_API_KEY` for the live model.")
        st.button("Use the demo question", on_click=lambda: st.session_state.update(pending=session.DEMO_QUESTION))
    else:
        st.success(f"Live: {config.LLM_MODEL}")
    if st.button("Clear conversation"):
        st.session_state.runs = []
    runs = st.session_state.setdefault("runs", [])
    st.caption(f"Tokens this session: {sum(r.usage['input_tokens'] + r.usage['output_tokens'] for r in runs)}")

question = st.chat_input("Ask about your properties") or st.session_state.pop("pending", None)
if question:
    registry, system = environment()
    history = runs[-1].messages if runs else None
    with st.spinner("Thinking..."):
        runs.append(run_logged(question, session.make_client(backend, question), registry, system,
                               history=history, model=config.LLM_MODEL if backend == "openai" else "fake"))


def show_message(conn, m) -> None:
    facts = invoice_facts(m["invoice_id"])
    who = f"{facts['name']} ({facts['unit']})" if facts else f"tenant {m['tenant_id']}"
    with st.container(border=True):
        st.markdown(f"**#{m['id']} · {who}** · {m['language'].upper()} · invoice {m['invoice_id']} · :blue[{m['status']}]")
        if m["status"] == "DRAFT":
            body = st.text_area("Message", m["body"], key=f"body-{m['id']}", height=100)
            reason = st.text_input("Reason (for a rejection, or a note on an edit)", key=f"why-{m['id']}")
            a, b, c = st.columns(3)
            try:
                if a.button("Approve", key=f"ok-{m['id']}", disabled=body != m["body"],
                            help="Approve the text as drafted. Edit the text to use Save edit instead."):
                    outbox.approve(conn, m["id"], facts)
                    st.rerun()
                if b.button("Save edit", key=f"ed-{m['id']}", disabled=body == m["body"]):
                    outbox.edit(conn, m["id"], body, facts, reason or None)
                    st.rerun()
                if c.button("Reject", key=f"no-{m['id']}"):
                    outbox.reject(conn, m["id"], reason or None)
                    st.rerun()
            except outbox.OutboxError as exc:
                st.error(str(exc))
        else:
            st.write(m["edited_body"] or m["body"])
            if m["reason"]:
                st.caption(f"Note: {m['reason']}")
            if m["status"] in ("APPROVED", "EDITED"):
                if st.button("Send (simulated)", key=f"send-{m['id']}", type="primary"):
                    try:
                        st.session_state["sent_note"] = outbox.send(conn, m["id"])
                    except outbox.OutboxError as exc:
                        st.session_state["sent_note"] = f"Not sent: {exc}"
                    st.rerun()
            elif m["status"] == "SENT":
                st.caption(f"Sent (simulated) {m['sent_at']}. Nothing left this machine.")


def outbox_tab() -> None:
    st.caption("The agent only drafts. Nothing is sent until you approve it, and sending here is simulated.")
    registry, _ = environment()
    ro = db.connect_ro()
    try:
        overdue = ro.execute("SELECT v.invoice_id, t.name, u.code, v.balance, v.days_late FROM v_arrears v "
                             "JOIN tenant t ON t.id = v.tenant_id JOIN unit u ON u.id = v.unit_id "
                             "ORDER BY v.days_late DESC LIMIT 30").fetchall()
        overdue = [dict(r) for r in overdue]
    finally:
        ro.close()
    pick = st.selectbox("Draft a reminder for an overdue invoice", overdue, index=None, placeholder="Choose an invoice",
                        format_func=lambda r: f"{r['name']} · {r['code']} · KES {r['balance']:,} · {r['days_late']} days late")
    if pick and st.button("Draft reminder"):
        res = registry.call("draft_reminder", {"invoice_id": pick["invoice_id"]})
        st.error(res["error"]) if "error" in res else st.success(f"Draft #{res['outbox_id']} saved for your review.")
    if note := st.session_state.pop("sent_note", None):
        st.info(note)
    conn = outbox.open_outbox(db.connect_rw())
    try:
        messages = outbox.listing(conn)
        shown = st.radio("Show", ["Needs review", "Ready to send", "All"], horizontal=True)
        keep = {"Needs review": ("DRAFT",), "Ready to send": ("APPROVED", "EDITED")}.get(shown)
        shown_msgs = [m for m in messages if keep is None or m["status"] in keep]
        if not shown_msgs:
            st.write("Nothing here.")
        for m in shown_msgs:
            show_message(conn, m)
    finally:
        conn.close()


ask, box = st.tabs(["Ask", "Outbox"])
with ask:
    if not runs:
        st.write(f"Try: *{session.DEMO_QUESTION}*")
    for n, run in enumerate(runs):
        show_run(run, str(n))
with box:
    outbox_tab()
