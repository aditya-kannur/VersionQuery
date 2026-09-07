"""
Multi-hop migration decomposition — breaks a from_version/to_version
range into an ordered sequence of known-version hops, then retrieves +
grades the migration chunk for each hop separately.
"""
from src.constants import KNOWN_VERSIONS


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


def decompose_and_retrieve(from_version, to_version, retrieve_and_grade_fn):
    """
    Builds the hop sequence, then retrieves + grades the migration chunk
    for each hop independently, using Day 8's retrieve_and_grade().
    retrieve_and_grade_fn should already be scoped to doc_type='migration'.
    """
    hops = get_hop_sequence(from_version, to_version)
    if hops is None:
        return {"status": "not_found"}

    hop_results = []
    for version_a, version_b in hops:
        question = f"What changed migrating from {version_a} to {version_b}?"
        result = retrieve_and_grade_fn(question, expected_version=version_b)

        if result["status"] != "found":
            # One hop failing grounds the whole migration answer as
            # incomplete/untrustworthy — don't silently skip a hop.
            return {"status": "not_found", "failed_hop": (version_a, version_b)}

        hop_results.append({
            "from": version_a,
            "to": version_b,
            "chunks": result["chunks"],
        })

    return {"status": "found", "hops": hop_results}