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

from src.messages import CLARIFICATION_QUESTION, NOT_FOUND_MESSAGE

API_URL = os.environ.get("VERSIONQUERY_API_URL", "http://localhost:8000")

st.set_page_config(page_title="VersionQuery", page_icon="VQ", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { --ink:#172033; --muted:#64748b; --line:#dbe4ef; --paper:#f8fafc; }
.stApp { background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans',sans-serif; }
.block-container { max-width:1120px; padding-top:3.5rem; padding-bottom:8rem; }
.hero { background:linear-gradient(135deg,#172033,#263b61); color:white; border-radius:24px; padding:2.5rem 3rem; box-shadow:0 18px 45px rgba(23,32,51,.16); }
.eyebrow { color:#93c5fd; font-family:'JetBrains Mono',monospace; font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { color:white; font-size:clamp(2.2rem,5vw,4.1rem); letter-spacing:-.06em; margin:.45rem 0 .65rem; }
.hero p { color:#dbeafe; font-size:1.05rem; max-width:670px; margin:0; line-height:1.6; }
.section-title { color:var(--ink); font-size:1.35rem; font-weight:700; margin:2rem 0 .8rem; }
.card { background:white; border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.25rem; min-height:115px; box-shadow:0 4px 14px rgba(23,32,51,.04); }
.card h3 { margin:.1rem 0 .45rem; color:var(--ink); font-size:1rem; }
.card p { color:var(--muted); line-height:1.5; margin:0; font-size:.93rem; }
.stChatMessage { border:1px solid var(--line); border-radius:16px; }
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
    for version in ("2021-05-13", "2021-08-16", "2022-02-22", "2022-06-28", "2025-09-03", "2026-03-11"):
        st.code(version, language=None)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Version-aware API documentation</div>
  <h1>Find the answer for the right version.</h1>
  <p>Ask about Notion API behavior, migrations, or release changes. VersionQuery retrieves matching documentation and shows where the answer came from.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-title">Start with a question</div>', unsafe_allow_html=True)
st.caption("Include a version when you can. For a general question, ask for the latest version and VersionQuery will resolve it automatically.")

cols = st.columns(3)
for col, title, body in zip(cols, ["Reference lookup", "Migration path", "Latest docs"], ["How do I query a database in 2022-06-28?", "What changed from 2021-08-16 to 2022-06-28?", "What is the latest API version in the docs?"]):
    with col:
        st.markdown(f'<div class="card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

with st.expander("How answers are produced"):
    st.markdown("1. Understand the intent and version.\n2. Search matching reference, migration, or changelog documents.\n3. Grade the evidence and generate a cited answer.\n4. Refuse unsupported claims instead of guessing.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn["answer"], turn.get("citations", []))

question = st.chat_input("Ask about a version, migration, or the latest API docs…")

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
