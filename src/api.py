"""
Day 11 stub, wired up on Day 13: FastAPI /ask route.

Builds the retrieval index once at process startup (not per-request) via
FastAPI's lifespan handler, then routes every /ask call through the
compiled LangGraph app from src/graph.py.
"""
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("versionquery")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.graph import ask as run_graph
from src.graph import build_graph
from src.constants import KNOWN_VERSIONS
from src.messages import LATEST_VERSION_MESSAGE, SERVICE_UNAVAILABLE_MESSAGE
from src.retrieval_pipeline import (
    build_bm25_index,
    build_chroma_collection,
    load_all_chunks,
)
from src.openrouter import embedding_function

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_all_chunks()
    collection = build_chroma_collection(chunks, embedding_function)
    bm25 = build_bm25_index(chunks)
    _state["app"] = build_graph(collection, bm25, chunks, embedding_function)
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

    normalized_question = question.lower()
    if "version" in normalized_question and any(
        term in normalized_question for term in ("latest", "newest", "most recent", "current")
    ):
        return AskResponse(
            answer=LATEST_VERSION_MESSAGE.format(version=KNOWN_VERSIONS[-1]),
            citations=[],
        )

    if "app" not in _state:
        # Startup's index build hasn't finished (or failed) — fail loudly
        # rather than crashing on a KeyError further down.
        raise HTTPException(status_code=503, detail="retrieval index is not ready yet")

    try:
        result = run_graph(_state["app"], question)
    except Exception:
        # Do not disguise provider failures as valid retrieval results.
        logger.exception("Pipeline failed for question: %r", question)
        return AskResponse(answer=SERVICE_UNAVAILABLE_MESSAGE, citations=[])

    return AskResponse(answer=result["answer"], citations=result["citations"])


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "app" in _state}
