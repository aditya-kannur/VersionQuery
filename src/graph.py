"""
Day 11 — wires query understanding -> routing -> retrieval/grading ->
(multi-hop decomposition for migration) -> generation -> verification into
a single LangGraph graph.

The graph doesn't build its own retrieval index; call build_graph() with an
already-built collection/bm25/chunks/embedding-model tuple (see
src/retrieval_pipeline.py) so the expensive index build happens once at
app startup, not per request.
"""
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END

from src.query_understanding import understand_query
from src.router import route_query
from src.grading import retrieve_and_grade
from src.migration_decomposition import decompose_and_retrieve
from src.generation import generate_grounded_answer
from src.messages import NOT_FOUND_MESSAGE
from src.retrieval_pipeline import hybrid_retrieve


class GraphState(TypedDict, total=False):
    question: str
    understood: Dict[str, Any]
    routing: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    requested_version: Optional[str]
    answer: str
    citations: List[Dict[str, Any]]


def build_graph(collection, bm25, chunks, embed_model):
    """
    Returns a compiled LangGraph app. `collection`, `bm25`, `chunks` and
    `embed_model` are the objects retrieval_pipeline.py's
    build_chroma_collection() / build_bm25_index() / load_all_chunks()
    produce — passed in rather than rebuilt so one process reuses one index.
    """

    def make_retrieve_fn(doc_type, version):
        # A retrieve_fn closure bound to one doc_type/version pair, in the
        # shape src.grading.retrieve_and_grade() expects: callable(question).
        def retrieve_fn(question_text):
            return hybrid_retrieve(
                question_text, collection, bm25, chunks, embed_model,
                doc_type=doc_type, version=version,
            )
        return retrieve_fn

    def node_understand(state: GraphState) -> dict:
        understood = understand_query(state["question"])
        return {"understood": understood}

    def node_route(state: GraphState) -> dict:
        routing = route_query(state["understood"])
        return {"routing": routing}

    def route_branch(state: GraphState) -> str:
        status = state["routing"]["status"]
        if status == "ready":
            intent = state["understood"].get("intent")
            return "retrieve_migration" if intent == "migration" else "retrieve_single"
        return "early_exit"

    def node_retrieve_single(state: GraphState) -> dict:
        routing = state["routing"]
        question = state["question"]
        version = routing.get("version")

        # Diagnostic searches changelog + reference independently (PRD
        # section 4.3) rather than as one pooled doc_type — a chunk that
        # passes grading from either search is usable evidence.
        all_passed = []
        for doc_type in routing["doc_types"]:
            result = retrieve_and_grade(
                question, expected_version=version, expected_doc_type=doc_type,
                retrieve_fn=make_retrieve_fn(doc_type, version),
            )
            if result["status"] == "found":
                all_passed.extend(result["chunks"])

        return {"chunks": all_passed, "requested_version": version}

    def node_retrieve_migration(state: GraphState) -> dict:
        routing = state["routing"]
        from_version = routing["from_version"]
        to_version = routing["to_version"]

        def retrieve_and_grade_fn(question_text, expected_version):
            return retrieve_and_grade(
                question_text, expected_version=expected_version, expected_doc_type="migration",
                retrieve_fn=make_retrieve_fn("migration", expected_version),
            )

        result = decompose_and_retrieve(from_version, to_version, retrieve_and_grade_fn)

        if result["status"] != "found":
            return {"chunks": [], "requested_version": f"{from_version} -> {to_version}"}

        # Flatten every hop's chunks into one list for generation — the
        # citation for each still carries its own hop's version.
        hop_chunks = [c for hop in result["hops"] for c in hop["chunks"]]
        return {"chunks": hop_chunks, "requested_version": f"{from_version} -> {to_version}"}

    def node_generate(state: GraphState) -> dict:
        result = generate_grounded_answer(
            state["question"], state["chunks"], state.get("requested_version"),
        )
        return {"answer": result["answer"], "citations": result["citations"]}

    def node_early_exit(state: GraphState) -> dict:
        # needs_clarification or not_found — routing already produced the
        # exact message to show, so generation never runs.
        return {
            "answer": state["routing"].get("message", NOT_FOUND_MESSAGE),
            "citations": [],
        }

    graph = StateGraph(GraphState)
    graph.add_node("understand", node_understand)
    graph.add_node("route", node_route)
    graph.add_node("retrieve_single", node_retrieve_single)
    graph.add_node("retrieve_migration", node_retrieve_migration)
    graph.add_node("generate", node_generate)
    graph.add_node("early_exit", node_early_exit)

    graph.set_entry_point("understand")
    graph.add_edge("understand", "route")
    graph.add_conditional_edges("route", route_branch, {
        "retrieve_single": "retrieve_single",
        "retrieve_migration": "retrieve_migration",
        "early_exit": "early_exit",
    })
    graph.add_edge("retrieve_single", "generate")
    graph.add_edge("retrieve_migration", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("early_exit", END)

    return graph.compile()


def ask(app, question: str) -> dict:
    """Runs one question through the compiled graph, returns {answer, citations}."""
    final_state = app.invoke({"question": question})
    return {"answer": final_state["answer"], "citations": final_state.get("citations", [])}
