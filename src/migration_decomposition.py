"""
Multi-hop migration decomposition — breaks a from_version/to_version
range into an ordered sequence of known-version hops, then retrieves +
grades the migration chunk for each hop separately.
"""
from src.constants import KNOWN_VERSIONS


def _has_topic_terms(question):
    if not question:
        return False

    stopwords = {
        "what", "changed", "changes", "change", "from", "to", "move",
        "moving", "migrate", "migration", "upgrade", "upgrading", "for",
        "my", "the", "a", "an", "api", "version", "notion", "need",
        "make", "i", "do", "does", "did",
    }
    terms = [
        term for term in question.lower().replace("-", " ").split()
        if len(term) > 2 and term not in stopwords and not term.isdigit()
    ]
    return bool(terms)


def get_hop_sequence(from_version, to_version):
    """
    Returns the ordered list of intermediate hops between from_version and
    to_version, using ONLY versions present in KNOWN_VERSIONS — never
    invents a version that doesn't exist in the dataset.
    """
    if from_version not in KNOWN_VERSIONS or to_version not in KNOWN_VERSIONS:
        # Either endpoint isn't a real known version — can't build a path.
        return None

    sorted_versions = sorted(KNOWN_VERSIONS)  # version strings are YYYY-MM-DD, so lexical sort == chronological sort
    start_idx = sorted_versions.index(from_version)
    end_idx = sorted_versions.index(to_version)

    if start_idx >= end_idx:
        # from_version isn't actually before to_version — invalid range.
        return None

    # Each hop is a (version_a, version_b) pair — the guide covering that
    # specific transition. e.g. [(2021-08-16, 2022-02-22), (2022-02-22, 2022-06-28)]
    hops = list(zip(sorted_versions[start_idx:end_idx], sorted_versions[start_idx + 1:end_idx + 1]))
    return hops


def decompose_and_retrieve(from_version, to_version, retrieve_and_grade_fn, original_question=None):
    """
    Builds the hop sequence, then retrieves + grades the migration chunk
    for each hop independently, using Day 8's retrieve_and_grade().
    retrieve_and_grade_fn should already be scoped to doc_type='migration'.

    original_question is the user's actual question. When provided it is
    passed to retrieve_and_grade so that grading scores relevance against
    what was really asked (e.g. "what changed in the Users GET route")
    rather than the generic per-hop stub. Falls back to the stub when the
    caller doesn't supply one (e.g. tests).
    """
    hops = get_hop_sequence(from_version, to_version)
    if hops is None:
        return {"status": "not_found"}

    hop_results = []
    use_topic_question = _has_topic_terms(original_question)
    for version_a, version_b in hops:
        # Use the user's question when available so grading can judge
        # relevance correctly; fall back to the generic stub otherwise.
        hop_stub = f"What changed migrating from {version_a} to {version_b}?"
        question = original_question if use_topic_question else hop_stub

        result = retrieve_and_grade_fn(question, expected_version=version_b)

        if result["status"] != "found":
            # A hop may have no relevant changes for a topic-specific
            # question. Keep collecting other grounded hops and only return
            # not_found if the whole requested range has no evidence.
            continue

        hop_results.append({
            "from": version_a,
            "to": version_b,
            "chunks": result["chunks"],
        })

    if not hop_results:
        return {"status": "not_found"}

    return {"status": "found", "hops": hop_results}
