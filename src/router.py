"""
Day 7: Routing logic — maps intent -> doc_type(s) + version filter,
and handles the clarification flow when version is missing.
"""
from src.constants import KNOWN_VERSIONS
from src.messages import CLARIFICATION_QUESTION, INVALID_VERSION_MESSAGE, NOT_FOUND_MESSAGE

# Maps each intent to which doc_type(s) it should search.
# Diagnostic searches two doc types in parallel per PRD section 4.3.
INTENT_TO_DOC_TYPES = {
    "reference": ["reference"],
    "diagnostic": ["changelog", "reference"],
    "migration": ["migration"],
}


def route_query(understood_query: dict) -> dict:
    """
    Takes the dict from query_understanding.understand_query() and turns it
    into a routing decision: which doc_type(s) to search, and what version
    filter(s) to apply. Returns a dict with a 'status' field so the caller
    knows whether to proceed to retrieval or ask for clarification.
    """
    intent = understood_query.get("intent")
    doc_types = INTENT_TO_DOC_TYPES.get(intent)

    if doc_types is None:
        # Intent itself wasn't one of the three known types — treat as
        # not-found rather than guessing a doc_type to search.
        return {"status": "not_found", "message": NOT_FOUND_MESSAGE}

    if intent == "migration":
        from_version = understood_query.get("from_version")
        to_version = understood_query.get("to_version")

        # Migration needs BOTH ends of the range — missing either one
        # means we can't build a hop sequence, so ask for clarification.
        if not from_version or not to_version:
            return {"status": "needs_clarification", "message": CLARIFICATION_QUESTION}

        invalid_versions = [
            version for version in (from_version, to_version)
            if version not in KNOWN_VERSIONS
        ]
        if invalid_versions:
            return {
                "status": "not_found",
                "message": INVALID_VERSION_MESSAGE.format(
                    version=", ".join(invalid_versions),
                    supported_versions=", ".join(KNOWN_VERSIONS),
                ),
            }

        return {
            "status": "ready",
            "doc_types": doc_types,
            "from_version": from_version,
            "to_version": to_version,
        }

    # reference / diagnostic path — single version field
    version = understood_query.get("version")
    if not version:
        # Per PRD 4.2: version missing and can't be inferred -> ask, don't guess.
        return {"status": "needs_clarification", "message": CLARIFICATION_QUESTION}

    if version not in KNOWN_VERSIONS:
        return {
            "status": "not_found",
            "message": INVALID_VERSION_MESSAGE.format(
                version=version,
                supported_versions=", ".join(KNOWN_VERSIONS),
            ),
        }

    return {
        "status": "ready",
        "doc_types": doc_types,
        "version": version,
    }
