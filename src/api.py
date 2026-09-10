"""
Day 11: FastAPI endpoint stub — /ask route.
Boilerplate only. src/graph.py supplies the actual pipeline logic.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="VersionQuery API")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    raise NotImplementedError("wire this up to src.graph.ask() once the index is built at startup")
