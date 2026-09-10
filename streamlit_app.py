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

st.set_page_config(page_title="VersionQuery", page_icon="📚")


def render_answer(answer: str, citations: list):
    """
    Distinct visual treatment per PRD state: clarification is a prompt for
    more input (info), not-found is an explicit honest failure (warning),
    a normal answer gets its citations displayed underneath.
    """
    if answer == CLARIFICATION_QUESTION:
        st.info(answer)
    elif answer == NOT_FOUND_MESSAGE:
        st.warning(answer)
    else:
        st.write(answer)
        for citation in citations:
            st.caption(citation["text"].replace("\n", " · "))


st.title("VersionQuery")
st.caption("Ask a question about the Notion API, scoped to a specific version.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn["answer"], turn.get("citations", []))

question = st.chat_input("e.g. How do I query a database in Notion-Version 2022-06-28?")

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
