"""
Day 11 stub, wired up on Day 13: FastAPI /ask route.

Builds the retrieval index once at process startup (not per-request) via
FastAPI's lifespan handler, then routes every /ask call through the
compiled LangGraph app from src/graph.py.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.chroma_config import EMBEDDING_MODEL_NAME
from src.graph import ask as run_graph
from src.graph import build_graph
from src.messages import NOT_FOUND_MESSAGE
from src.retrieval_pipeline import (
    build_bm25_index,
    build_chroma_collection,
    load_all_chunks,
)

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_all_chunks()
    collection = build_chroma_collection(chunks)
    bm25 = build_bm25_index(chunks)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _state["app"] = build_graph(collection, bm25, chunks, embed_model)
    yield
    _state.clear()


app = FastAPI(title="VersionQuery API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if "app" not in _state:
        # Startup's index build hasn't finished (or failed) — fail loudly
        # rather than crashing on a KeyError further down.
        raise HTTPException(status_code=503, detail="retrieval index is not ready yet")

    try:
        result = run_graph(_state["app"], question)
    except Exception:
        # An LLM call failing mid-pipeline (rate limit, malformed JSON,
        # network blip) should degrade to the same honest-failure message
        # a "no grounding found" case gets — never a raw 500 with a stack
        # trace, and never a silently wrong answer.
        return AskResponse(answer=NOT_FOUND_MESSAGE, citations=[])

    return AskResponse(answer=result["answer"], citations=result["citations"])


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "app" in _state}
