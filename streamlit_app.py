"""
Day 12 — Streamlit chat UI for VersionQuery.
Input box, conversation history, citation display for each answer.

Builds the retrieval index once at startup (st.cache_resource), then runs
every question through the LangGraph app from src/graph.py.
"""
import streamlit as st

from src.graph import build_graph, ask
from src.retrieval_pipeline import load_all_chunks, build_chroma_collection, build_bm25_index
from src.chroma_config import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="VersionQuery", page_icon="📚")


@st.cache_resource(show_spinner="Building the retrieval index (first run only)...")
def get_app():
    chunks = load_all_chunks()
    collection = build_chroma_collection(chunks)
    bm25 = build_bm25_index(chunks)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return build_graph(collection, bm25, chunks, embed_model)


st.title("VersionQuery")
st.caption("Ask a question about the Notion API, scoped to a specific version.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        for citation in turn.get("citations", []):
            st.caption(citation["text"].replace("\n", " · "))

question = st.chat_input("e.g. How do I query a database in Notion-Version 2022-06-28?")

if question:
    st.session_state.history.append({"question": question, "answer": None, "citations": []})
    with st.chat_message("user"):
        st.write(question)

    app = get_app()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(app, question)
        st.write(result["answer"])
        for citation in result["citations"]:
            st.caption(citation["text"].replace("\n", " · "))

    st.session_state.history[-1]["answer"] = result["answer"]
    st.session_state.history[-1]["citations"] = result["citations"]
