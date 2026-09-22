"""
Query understanding — classifies intent (reference/diagnostic/migration)
and extracts version(s) from the developer's question, using Gemini.
"""
import json

from src.prompts.system_prompt import SYSTEM_PROMPT   # teammate's Day 6 file
from src.constants import KNOWN_VERSIONS               # Day 2 teammate file

from src.openrouter import generate_text


def understand_query(user_question: str) -> dict:
    """
    Returns a dict like:
      {"intent": "reference", "version": "2022-06-28"}
      {"intent": "migration", "from_version": "2021-08-16", "to_version": "2022-06-28"}
      {"intent": "reference", "version": None}   # version missing -> triggers clarification (Day 7)
    """
    prompt = f"""{SYSTEM_PROMPT}

Known versions: {", ".join(KNOWN_VERSIONS)}

Developer question: "{user_question}"

Respond with ONLY valid JSON, no markdown fences, no explanation.
"""

    raw_text = generate_text(prompt).strip()

    # Gemini sometimes wraps JSON in ```json fences even when told not to —
    # strip them defensively before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").removeprefix("json").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fail loud, not silent — a malformed LLM response shouldn't
        # masquerade as a valid "no version found" result.
        raise ValueError(f"Could not parse model output as JSON: {raw_text}")

    if not result.get("version") and any(
        term in user_question.lower() for term in ("latest", "newest", "most recent", "current")
    ):
        result["version"] = KNOWN_VERSIONS[-1]

    return result
