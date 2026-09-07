"""
LLM-based multi-dimensional document grading — checks each
retrieved chunk for relevance, version correctness, and doc-type fit.
Failed grading triggers a query-rewrite retry, capped per config.
"""
import json
import google.generativeai as genai

from src.grading_config import MAX_RETRIES, RELEVANCE_THRESHOLD  # teammate's Day 8 file
from src.messages import NOT_FOUND_MESSAGE

model = genai.GenerativeModel("gemini-1.5-flash")

GRADING_PROMPT = """You are grading whether a retrieved document chunk actually
answers a developer's question.

Question: "{question}"
Expected version: {expected_version}
Expected doc type: {expected_doc_type}

Chunk metadata: {chunk_metadata}
Chunk text: "{chunk_text}"

Score the chunk on three dimensions, each true/false:
- "relevant": does the chunk's content actually address the question?
- "version_correct": does the chunk's version match the expected version?
- "doc_type_correct": does the chunk's doc_type match the expected doc type?

Respond with ONLY valid JSON: {{"relevant": bool, "version_correct": bool, "doc_type_correct": bool}}
"""


def grade_chunk(question, expected_version, expected_doc_type, chunk):
    """
    Grades a single chunk. Returns True only if it passes all 3 dimensions.
    """
    prompt = GRADING_PROMPT.format(
        question=question,
        expected_version=expected_version,
        expected_doc_type=expected_doc_type,
        chunk_metadata=json.dumps({k: v for k, v in chunk.items() if k != "text"}),
        chunk_text=chunk["text"],
    )

    response = model.generate_content(prompt)
    raw = response.text.strip().strip("`").removeprefix("json").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # A grading call we can't parse should fail the chunk, not pass it —
        # never let an ungraded chunk slip through as "good."
        return False

    return result.get("relevant") and result.get("version_correct") and result.get("doc_type_correct")


def grade_chunks(question, expected_version, expected_doc_type, chunks):
    """Returns only the chunks that pass grading on all 3 dimensions."""
    return [
        c for c in chunks
        if grade_chunk(question, expected_version, expected_doc_type, c)
    ]


def rewrite_query(original_question):
    """
    Asks the model to rephrase the question for a retry — e.g. rewording
    ambiguous phrasing, expanding abbreviations — without changing intent.
    """
    prompt = f"""Rewrite this developer question to be clearer and more specific
for a document search, without changing its meaning or intent:

"{original_question}"

Respond with ONLY the rewritten question, nothing else."""
    response = model.generate_content(prompt)
    return response.text.strip()


def retrieve_and_grade(question, expected_version, expected_doc_type, retrieve_fn):
    """
    The full retry loop: retrieve -> grade -> if nothing passes, rewrite
    query and retry, capped at MAX_RETRIES total attempts.
    retrieve_fn is a callable like hybrid_retrieve bound to the right args.
    """
    current_question = question

    for attempt in range(MAX_RETRIES + 1):  # +1 because attempt 0 is the original try
        chunks = retrieve_fn(current_question)
        passed = grade_chunks(current_question, expected_version, expected_doc_type, chunks)

        if passed:
            return {"status": "found", "chunks": passed}

        if attempt < MAX_RETRIES:
            current_question = rewrite_query(current_question)  # only rewrite if we still have attempts left

    # Exhausted all retries with nothing passing grading — honest failure.
    return {"status": "not_found", "message": NOT_FOUND_MESSAGE}