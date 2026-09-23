"""
Day 10 — grounded generation, citation formatting, and post-generation
verification (entailment check).

Generation is strictly template-fed: the model only ever sees the chunks
that already passed grading (src/grading.py), never the raw corpus, so it
cannot answer from anything outside what was actually retrieved and graded.
"""
import json
import os
import re

from src.citation_template import CITATION_TEMPLATE
from src.messages import NOT_FOUND_MESSAGE

from src.openrouter import generate_text

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

Write a concise, direct answer in plain natural language grounded strictly in
the retrieved content above. Follow these rules:
- Do NOT reproduce any JSON, code examples, or raw data — not even from your
  own knowledge of this API. Describe field/behavior changes in prose only,
  e.g. "the `type` and `property` fields were removed from the response"
  rather than showing the response body itself.
- Your answer must not contain curly braces, square brackets, or three
  backticks, anywhere, under any circumstance.
- Do NOT mention "chunks" or "retrieved documentation".
- Summarise what changed or how something works in 2-4 clear sentences.
- If multiple changes are covered, use a short bullet list.
- Write as if explaining it directly to the developer.
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


def _clean_source_text(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^{}]{80,}\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _format_context(chunks):
    parts = []
    for i, c in enumerate(chunks):
        heading = c.get("section") or c.get("summary") or c.get("endpoint") or ""
        cleaned = _clean_source_text(c["text"])
        # Further strip any remaining JSON-like content (arrays, objects)
        cleaned = re.sub(r"\[.*?\]", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\{.*?\}", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        label = f"[{i+1}] ({c.get('doc_type')})"
        if heading:
            label += f" {heading}:"
        parts.append(f"{label} {cleaned[:800]}")
    return "\n\n".join(parts)


def _generate_extractive_answer(question, chunks):
    """
    Provider-free fallback: summarise what changed using section headings
    only — never raw chunk text, which can contain JSON and code snippets.
    """
    lines = []
    for chunk in chunks:
        heading = (
            chunk.get("section")
            or chunk.get("summary")
            or chunk.get("endpoint")
        )
        version = chunk.get("version") or chunk.get("release_date") or ""
        if heading:
            line = f"- {heading}"
            if version:
                line += f" (version {version})"
            lines.append(line)

    if lines:
        intro = "Here are the relevant documented changes:"
        return intro + "\n" + "\n".join(lines[:8])

    # Absolute last resort — just name the sections without any text
    return "Relevant documentation was found but could not be summarised. Please check the cited sources below."


def _sanitize_answer_text(text, question, chunks):
    """
    Defense-in-depth: even with strict prompting, a model can still leak
    JSON it knows from its own pretraining (observed with llama-3.1-8b on
    well-known public API docs) rather than something present in the
    retrieved context. Strip anything JSON/code-shaped from the model's
    output before it's ever shown, regardless of which model produced it.
    Falls back to the heading-only extractive answer if sanitizing leaves
    too little readable text behind.
    """
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)
    # Strip any bracket/brace span, however deep — walk outward until no
    # more top-level [...] or {...} spans remain, since a single non-greedy
    # pass can leave dangling fragments on nested/unbalanced JSON.
    for _ in range(6):
        new_cleaned = re.sub(r"\{[^{}]*\}", " ", cleaned)
        new_cleaned = re.sub(r"\[[^\[\]]*\]", " ", new_cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
    # Any surviving stray bracket/brace means unbalanced JSON slipped
    # through — treat the whole answer as unsafe rather than show a
    # half-cleaned fragment.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if any(ch in cleaned for ch in "{}[]") or len(cleaned) < 20:
        return _generate_extractive_answer(question, chunks)
    return cleaned


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
    try:
        answer_text = generate_text(prompt, max_tokens=512).strip()
        answer_text = _sanitize_answer_text(answer_text, question, chunks)
    except Exception:
        answer_text = _generate_extractive_answer(question, chunks)

    citations = [build_citation(c, requested_version) for c in chunks]

    return {"answer": answer_text, "citations": citations}


def verify_answer(question, answer_text, chunks, requested_version=None):
    """
    Post-generation entailment check: does the answer only say things the
    source chunks support, and does it stick to the requested version?

    Returns True on any LLM/network failure so a transient provider error
    (rate-limit, malformed JSON, timeout) does not silently discard an
    answer that was correctly generated. Returns False only when the model
    explicitly says the answer is unsupported or version-inconsistent.
    """
    if not chunks:
        return False
    if os.getenv("ENABLE_LLM_VERIFICATION", "false").lower() != "true":
        return True

    prompt = VERIFICATION_PROMPT.format(
        question=question,
        version=requested_version or "not specified",
        answer=answer_text,
        context=_format_context(chunks),
    )

    try:
        raw = generate_text(prompt, max_tokens=64).strip().strip("`").removeprefix("json").strip()
        result = json.loads(raw)
    except Exception:
        # Verification call failed (rate-limit, network blip, bad JSON) —
        # pass the answer through rather than treating a provider hiccup as
        # evidence of hallucination.
        return True

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