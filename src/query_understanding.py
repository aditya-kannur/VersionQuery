"""
Query understanding — classifies intent (reference/diagnostic/migration)
and extracts version(s) from the developer's question, using Gemini.
"""
import os
import json
import google.generativeai as genai

from src.prompts.system_prompt import SYSTEM_PROMPT   # teammate's Day 6 file
from src.constants import KNOWN_VERSIONS               # Day 2 teammate file

genai.configure(api_key=os.environ["GEMINI_API_KEY"])   # reads your env var, never hardcode the key
model = genai.GenerativeModel("gemini-1.5-flash")        # fast/cheap model — fine for classification, not generation


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

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Gemini sometimes wraps JSON in ```json fences even when told not to —
    # strip them defensively before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").removeprefix("json").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fail loud, not silent — a malformed LLM response shouldn't
        # masquerade as a valid "no version found" result.
        raise ValueError(f"Could not parse model output as JSON: {raw_text}")