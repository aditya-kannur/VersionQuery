"""
Day 15 — edge-case tests for the pure (non-LLM) logic in the pipeline.
Deliberately excludes query_understanding.py / grading.py / generation.py,
which all call Gemini and need GEMINI_API_KEY — those are exercised live
by src/evaluate.py instead.
"""
from src.router import route_query
from src.migration_decomposition import get_hop_sequence
from src.generation import build_citation
from src.messages import CLARIFICATION_QUESTION, NOT_FOUND_MESSAGE


# --- router.py --------------------------------------------------------

def test_route_unknown_intent_is_not_found():
    result = route_query({"intent": "bogus"})
    assert result["status"] == "not_found"
    assert result["message"] == NOT_FOUND_MESSAGE


def test_route_reference_missing_version_asks_for_clarification():
    result = route_query({"intent": "reference", "version": None})
    assert result["status"] == "needs_clarification"
    assert result["message"] == CLARIFICATION_QUESTION


def test_route_migration_missing_to_version_asks_for_clarification():
    result = route_query({"intent": "migration", "from_version": "2021-08-16", "to_version": None})
    assert result["status"] == "needs_clarification"


def test_route_diagnostic_searches_two_doc_types():
    result = route_query({"intent": "diagnostic", "version": "2022-06-28"})
    assert result["status"] == "ready"
    assert set(result["doc_types"]) == {"changelog", "reference"}


# --- migration_decomposition.py ---------------------------------------

def test_hop_sequence_reversed_range_is_invalid():
    assert get_hop_sequence("2022-06-28", "2021-08-16") is None


def test_hop_sequence_identical_versions_is_invalid():
    assert get_hop_sequence("2022-06-28", "2022-06-28") is None


def test_hop_sequence_unknown_version_is_invalid():
    assert get_hop_sequence("2020-01-01", "2022-06-28") is None


def test_hop_sequence_never_invents_an_intermediate_version():
    hops = get_hop_sequence("2021-05-13", "2026-03-11")
    all_versions_in_hops = {v for pair in hops for v in pair}
    from src.constants import KNOWN_VERSIONS
    assert all_versions_in_hops <= set(KNOWN_VERSIONS)


# --- generation.py: citation building ----------------------------------

def test_citation_reference_uses_requested_version_not_the_full_list():
    chunk = {"doc_type": "reference", "summary": "Query a database", "versions": ["2025-09-03", "2026-03-11"]}
    citation = build_citation(chunk, requested_version="2026-03-11")
    assert citation["version"] == "2026-03-11"


def test_citation_changelog_missing_section_falls_back():
    chunk = {"doc_type": "changelog", "section": None, "release_date": "August 13, 2026"}
    citation = build_citation(chunk)
    assert citation["section"] == "Release notes"
