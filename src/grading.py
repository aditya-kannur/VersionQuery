"""
LLM-based multi-dimensional document grading — checks each
retrieved chunk for relevance, version correctness, and doc-type fit.
Failed grading triggers a query-rewrite retry, capped per config.
"""
import json
import os
import re
import time

from src.grading_config import MAX_RETRIES, RELEVANCE_THRESHOLD  # teammate's Day 8 file
from src.messages import NOT_FOUND_MESSAGE

from src.openrouter import generate_text

# Small pause between consecutive per-chunk grading calls so we don't
# burst all 3 calls at once and immediately saturate the free-tier
# per-minute quota. 1 second is enough to spread them out without
# noticeably slowing down a single request.
_INTER_CHUNK_SLEEP = float(os.getenv("LLM_GRADING_SLEEP_SECONDS", "0"))

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
    if os.getenv("ENABLE_LLM_GRADING", "false").lower() != "true":
        return _grade_chunk_locally(question, expected_version, expected_doc_type, chunk)

    prompt = GRADING_PROMPT.format(
        question=question,
        expected_version=expected_version,
        expected_doc_type=expected_doc_type,
        chunk_metadata=json.dumps({k: v for k, v in chunk.items() if k != "text"}),
        chunk_text=chunk["text"],
    )

    try:
        raw = generate_text(prompt, max_tokens=64).strip().strip("`").removeprefix("json").strip()
    except Exception:
        return _grade_chunk_locally(question, expected_version, expected_doc_type, chunk)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # A grading call we can't parse should fail the chunk, not pass it —
        # never let an ungraded chunk slip through as "good."
        return False

    return result.get("relevant") and result.get("version_correct") and result.get("doc_type_correct")


def _grade_chunk_locally(question, expected_version, expected_doc_type, chunk):
    """Lexical fallback used when no LLM provider is configured."""
    if chunk.get("doc_type") != expected_doc_type:
        return False

    if expected_version is not None:
        if "versions" in chunk:
            if expected_version not in chunk["versions"]:
                return False
        elif chunk.get("version") != expected_version:
            return False

    stopwords = {
        "the", "a", "an", "and", "or", "to", "do", "i", "in", "for", "of",
        "is", "my", "how", "what", "why", "does", "did", "api", "version",
        "notion", "from", "changed", "changes", "change", "moving", "move",
        "migrate", "migration", "upgrade", "upgrading", "need", "make",
    }
    question_terms = {
        term for term in re.findall(r"[a-z0-9_]+", question.lower())
        if (
            len(term) > 2
            and term not in stopwords
            and not re.fullmatch(r"\d+", term)
            and not re.fullmatch(r"20\d{2}", term)
        )
    }
    expanded_terms = set()
    for term in question_terms:
        expanded_terms.add(term)
        if term.endswith("ies") and len(term) > 4:
            expanded_terms.add(term[:-3] + "y")
        if term.endswith("s") and len(term) > 3:
            expanded_terms.add(term[:-1])
    question_terms = expanded_terms
    if not question_terms:
        return True

    searchable = " ".join(
        str(chunk.get(field, ""))
        for field in ("text", "summary", "endpoint", "section")
    ).lower()
    matches = sum(1 for term in question_terms if term in searchable)
    return matches >= max(1, min(2, len(question_terms)))


def grade_chunks(question, expected_version, expected_doc_type, chunks):
    """Returns only the chunks that pass grading on all 3 dimensions."""
    passed = []
    for i, c in enumerate(chunks):
        if i > 0:
            time.sleep(_INTER_CHUNK_SLEEP)
        if grade_chunk(question, expected_version, expected_doc_type, c):
            passed.append(c)
    return passed


def rewrite_query(original_question):
    """
    Asks the model to rephrase the question for a retry — e.g. rewording
    ambiguous phrasing, expanding abbreviations — without changing intent.
    """
    prompt = f"""Rewrite this developer question to be clearer and more specific
for a document search, without changing its meaning or intent:

"{original_question}"

Respond with ONLY the rewritten question, nothing else."""
    try:
        return generate_text(prompt, max_tokens=96).strip()
    except Exception:
        return original_question


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
