"""
Query understanding - classifies intent (reference/diagnostic/migration)
and extracts version(s) from the developer's question.
"""
import json
import re

from src.prompts.system_prompt import SYSTEM_PROMPT   # teammate's Day 6 file
from src.constants import KNOWN_VERSIONS               # Day 2 teammate file

from src.openrouter import generate_text


def _understand_query_locally(user_question: str) -> dict:
    """
    Provider-free fallback for local/dev use.

    It is intentionally conservative: extract explicit known versions, infer
    migration only when the question names a range/from-to pair, otherwise use
    simple intent keywords and let the router ask for clarification as needed.
    """
    normalized = user_question.lower()
    versions = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", user_question)
    known_versions = [version for version in versions if version in KNOWN_VERSIONS]
    primary_version = known_versions[0] if known_versions else (versions[0] if versions else None)

    latest_terms = ("latest", "newest", "most recent", "current")
    asks_for_latest_version = (
        "version" in normalized and any(term in normalized for term in latest_terms)
    )
    if asks_for_latest_version:
        return {
            "intent": "reference",
            "version": KNOWN_VERSIONS[-1],
            "latest_version_query": True,
        }

    migration_terms = ("migrate", "migration", "upgrade", "changed from", "from ")
    if any(term in normalized for term in migration_terms) and len(known_versions) >= 2:
        return {
            "intent": "migration",
            "from_version": known_versions[0],
            "to_version": known_versions[1],
        }

    diagnostic_terms = ("why", "break", "broken", "error", "empty", "worked yesterday", "changed")
    intent = "diagnostic" if any(term in normalized for term in diagnostic_terms) else "reference"
    return {"intent": intent, "version": primary_version}


def understand_query(user_question: str) -> dict:
    """
    Returns a dict like:
      {"intent": "reference", "version": "2022-06-28"}
      {"intent": "migration", "from_version": "2021-08-16", "to_version": "2022-06-28"}
      {"intent": "reference", "version": None}   # version missing -> triggers clarification (Day 7)
    """
    local_result = _understand_query_locally(user_question)
    if (
        local_result.get("latest_version_query")
        or local_result.get("intent") == "migration"
        or local_result.get("version")
    ):
        return local_result

    prompt = f"""{SYSTEM_PROMPT}

Known versions: {", ".join(KNOWN_VERSIONS)}

Developer question: "{user_question}"

Respond with ONLY valid JSON, no markdown fences, no explanation.
"""

    try:
        raw_text = generate_text(prompt, max_tokens=96).strip()
    except Exception:
        return _understand_query_locally(user_question)

    # Gemini sometimes wraps JSON in ```json fences even when told not to —
    # strip them defensively before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").removeprefix("json").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Malformed LLM output — we can't trust the classification, so
        # treat it as a missing-version clarification rather than crashing
        # the entire request. The user gets asked to rephrase instead of
        # seeing a 500 / SERVICE_UNAVAILABLE response.
        return {"intent": None, "version": None, "parse_error": True}

    latest_terms = ("latest", "newest", "most recent", "current")
    asks_for_latest_version = (
        "version" in user_question.lower()
        and any(term in user_question.lower() for term in latest_terms)
    )
    if not result.get("version") and asks_for_latest_version:
        result["version"] = KNOWN_VERSIONS[-1]

    if asks_for_latest_version:
        result["latest_version_query"] = True

    return result
