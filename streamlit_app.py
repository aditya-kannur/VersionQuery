"""
Day 12 UI, wired to the backend on Day 13 — Streamlit chat UI for
VersionQuery, talking to the FastAPI /ask endpoint over HTTP rather than
invoking the pipeline in-process, so the frontend and backend can be
deployed and scaled independently.

Run the backend first: uvicorn src.api:app
Then: streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

from src.messages import (
    CLARIFICATION_QUESTION,
    NOT_FOUND_MESSAGE,
    SERVICE_UNAVAILABLE_MESSAGE,
)

API_URL = os.environ.get("VERSIONQUERY_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="VersionQuery",
    page_icon="VQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { --ink:#172033; --muted:#64748b; --line:#dbe4ef; --paper:#f8fafc; }
.stApp { background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans',sans-serif; }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:8rem; }
.hero { background:linear-gradient(135deg,#172033,#263b61); color:white; border-radius:20px; padding:2rem 2.5rem; box-shadow:0 16px 38px rgba(23,32,51,.14); }
.eyebrow { color:#93c5fd; font-family:'JetBrains Mono',monospace; font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { color:white; font-size:clamp(2rem,4vw,3.15rem); letter-spacing:-.05em; margin:.45rem 0 .55rem; }
.hero p { color:#dbeafe; font-size:1rem; max-width:700px; margin:0; line-height:1.6; }
.status-strip { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }
.status-pill { background:rgba(219,234,254,.12); border:1px solid rgba(219,234,254,.22); border-radius:999px; color:#dbeafe; font-family:'JetBrains Mono',monospace; font-size:.75rem; padding:.35rem .6rem; }
.section-title { color:var(--ink); font-size:1.25rem; font-weight:700; margin:1.6rem 0 .45rem; }
.card { background:white; border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.25rem; min-height:110px; box-shadow:0 4px 14px rgba(23,32,51,.04); }
.card-title { margin:.1rem 0 .45rem; color:var(--ink); font-size:1rem; font-weight:700; }
.card p { color:var(--muted); line-height:1.5; margin:0; font-size:.93rem; }
.stButton > button { border-radius:10px; border:1px solid #cbd5e1; color:#1e3a5f; font-weight:600; transition:all .2s ease; }
.stButton > button:hover { border-color:#2563eb; color:#1d4ed8; background:#eff6ff; }
.stChatMessage { border:1px solid var(--line); border-radius:16px; }
@media (max-width: 700px) { .block-container { padding:1rem 1rem 7rem; } .hero { padding:1.5rem; } }
</style>
""", unsafe_allow_html=True)


def render_answer(answer: str, citations: list):
    """
    Distinct visual treatment per PRD state: clarification is a prompt for
    more input (info), not-found is an explicit honest failure (warning),
    a normal answer gets its citations displayed underneath.
    """
    if answer == CLARIFICATION_QUESTION:
        st.info(answer)
    elif answer == SERVICE_UNAVAILABLE_MESSAGE:
        st.error(answer)
    elif answer == NOT_FOUND_MESSAGE or answer.startswith("I couldn't find API version"):
        st.warning(answer)
    else:
        st.write(answer)
        for citation in citations:
            st.caption(citation["text"].replace("\n", " · "))


with st.sidebar:
    st.markdown("### VersionQuery")
    st.caption("Version-aware documentation search")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**What this tool does**")
    st.caption("Answers from versioned Notion API reference, migration, and changelog documents. It does not guess or use web search.")
    st.markdown("**Supported versions**")
    st.caption("2021-05-13 · 2021-08-16 · 2022-02-22 · 2022-06-28 · 2025-09-03 · 2026-03-11")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Version-aware API documentation</div>
  <h1>Versioned answers. No guessing.</h1>
  <p>Ask about Notion API behavior, migrations, or release changes. VersionQuery checks the matching documentation and returns the source with every grounded answer.</p>
  <div class="status-strip">
    <span class="status-pill">6 tracked versions</span>
    <span class="status-pill">Reference · migration · changelog</span>
    <span class="status-pill">Citations included</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-title">Try a documented path</div>', unsafe_allow_html=True)
st.caption("Use an example below or ask your own question. Include a version when you can; “latest API version” is resolved automatically.")

examples = [
    ("Reference lookup", "How do I retrieve a database in 2026-03-11?"),
    ("Migration path", "What changed from 2021-08-16 to 2022-06-28?"),
    ("Latest docs", "What is the latest API version in the docs?"),
]
cols = st.columns(3)
for index, (col, (title, body)) in enumerate(zip(cols, examples)):
    with col:
        st.markdown(f'<div class="card"><div class="card-title">{title}</div><p>{body}</p></div>', unsafe_allow_html=True)
        if st.button("Try this example", key=f"example_{index}", use_container_width=True):
            st.session_state.pending_question = body

with st.expander("How answers are produced"):
    st.markdown("1. Understand the intent and version.\n2. Search matching reference, migration, or changelog documents.\n3. Grade the evidence and generate a cited answer.\n4. Refuse unsupported claims instead of guessing.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn["answer"], turn.get("citations", []))

typed_question = st.chat_input("Ask about a version, migration, or the latest API docs…")
question = st.session_state.pop("pending_question", None) or typed_question

if question:
    st.session_state.history.append({"question": question, "answer": None, "citations": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Couldn't reach the VersionQuery API at {API_URL}: {exc}")
                result = {"answer": NOT_FOUND_MESSAGE, "citations": []}

        render_answer(result["answer"], result["citations"])

    st.session_state.history[-1]["answer"] = result["answer"]
    st.session_state.history[-1]["citations"] = result["citations"]
