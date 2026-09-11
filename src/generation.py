"""
Day 10 — grounded generation, citation formatting, and post-generation
verification (entailment check).

Generation is strictly template-fed: the model only ever sees the chunks
that already passed grading (src/grading.py), never the raw corpus, so it
cannot answer from anything outside what was actually retrieved and graded.
"""
import json

import google.generativeai as genai

from src.citation_template import CITATION_TEMPLATE
from src.messages import NOT_FOUND_MESSAGE

model = genai.GenerativeModel("gemini-3.5-flash-lite")  # gemini-1.5-flash was retired, gemini-2.5-flash-lite is closed to new users as of this key -- Google's own 404 named this as the replacement

# Human-readable document names per doc_type, for the citation block —
# matches the naming used in the PRD's example interaction.
DOCUMENT_NAMES = {
    "reference": "Notion API Reference",
    "migration": "Upgrade Guide",
    "changelog": "Changelog",
}

GENERATION_PROMPT = """You are answering a developer's question using ONLY the
retrieved documentation chunks below. Do not use general knowledge, and do not
state anything that isn't directly supported by these chunks.

Developer question: "{question}"
API version in question: {version}

Retrieved documentation:
{context}

Write a concise, direct answer grounded strictly in the retrieved content
above. Do not mention "chunks" or "retrieved documentation" in your answer —
write as if explaining it directly to the developer.
"""

VERIFICATION_PROMPT = """You are checking whether a generated answer is fully
supported by the source chunks it was generated from, and does not mix
content from a different API version than the one requested.

Question: "{question}"
Requested version: {version}
Generated answer: "{answer}"

Source chunks:
{context}

Respond with ONLY valid JSON: {{"supported": bool, "version_consistent": bool}}
"supported" is false if the answer states anything not present in the source
chunks. "version_consistent" is false if the answer mixes in behavior from a
version other than the one requested.
"""


def build_citation(chunk, requested_version=None):
    """
    Maps a chunk's doc_type-specific metadata onto the shared
    {document, section, version} citation shape, then formats it with
    CITATION_TEMPLATE.
    """
    doc_type = chunk.get("doc_type")
    document = DOCUMENT_NAMES.get(doc_type, doc_type or "Documentation")

    if doc_type == "reference":
        section = chunk.get("summary") or chunk.get("endpoint", "")
        # Reference chunks apply to more than one version at once — cite
        # the version the developer actually asked about, not the list.
        version = requested_version or ", ".join(chunk.get("versions", []))
    elif doc_type == "migration":
        section = chunk.get("section", "")
        version = chunk.get("version") or requested_version
    else:  # changelog
        section = chunk.get("section") or "Release notes"
        version = requested_version or chunk.get("release_date", "")

    return {
        "document": document,
        "section": section,
        "version": version,
        "text": CITATION_TEMPLATE.format(document=document, section=section, version=version),
    }


def _format_context(chunks):
    return "\n\n".join(
        f"[{i+1}] ({c.get('doc_type')}) {c['text']}" for i, c in enumerate(chunks)
    )


def generate_answer(question, chunks, requested_version=None):
    """
    Generates an answer strictly from the given (already-graded) chunks,
    then builds a citation for each one. Returns None chunks in ->
    NOT_FOUND_MESSAGE rather than calling the model on empty context.
    """
    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "citations": []}

    prompt = GENERATION_PROMPT.format(
        question=question,
        version=requested_version or "not specified",
        context=_format_context(chunks),
    )
    response = model.generate_content(prompt)
    answer_text = response.text.strip()

    citations = [build_citation(c, requested_version) for c in chunks]

    return {"answer": answer_text, "citations": citations}


def verify_answer(question, answer_text, chunks, requested_version=None):
    """
    Post-generation entailment check: does the answer only say things the
    source chunks support, and does it stick to the requested version?
    Returns False (fails closed) on any parse failure — an answer we can't
    verify should never be presented as verified.
    """
    if not chunks:
        return False

    prompt = VERIFICATION_PROMPT.format(
        question=question,
        version=requested_version or "not specified",
        answer=answer_text,
        context=_format_context(chunks),
    )
    response = model.generate_content(prompt)
    raw = response.text.strip().strip("`").removeprefix("json").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return False

    return bool(result.get("supported")) and bool(result.get("version_consistent"))


def generate_grounded_answer(question, chunks, requested_version=None):
    """
    Full Day 10 flow: generate -> verify -> only return the answer if it
    passes verification, otherwise fall back to the honest-failure message
    rather than showing an answer that failed its own entailment check.
    """
    result = generate_answer(question, chunks, requested_version)

    if result["answer"] == NOT_FOUND_MESSAGE:
        return result

    if not verify_answer(question, result["answer"], chunks, requested_version):
        return {"answer": NOT_FOUND_MESSAGE, "citations": []}

    return result
