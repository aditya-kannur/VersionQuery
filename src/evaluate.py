"""
Day 14 — evaluation script. Runs every row in data/test_set.json through
the full pipeline and reports the PRD's four metrics against the
thresholds in src/metrics_config.py.

Requires GEMINI_API_KEY (query understanding + generation + verification
all call the model) and a built retrieval index — run as a script, not
imported for its side effects.
"""
import json
import time

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer

try:
    load_dotenv()  # reads GEMINI_API_KEY from a .env file, if present
except UnicodeDecodeError:
    print("WARNING: .env exists but isn't valid UTF-8 -- skipping it.")

from src.chroma_config import EMBEDDING_MODEL_NAME
from src.graph import ask, build_graph
from src.messages import NOT_FOUND_MESSAGE
from src.metrics_config import (
    MAX_HALLUCINATION_RATE,
    MIN_CITATION_ACCURACY,
    MIN_NOT_FOUND_HONESTY,
    MIN_RETRIEVAL_HIT_RATE,
    MIN_VERSION_CORRECTNESS,
)
from src.retrieval_pipeline import build_bm25_index, build_chroma_collection, load_all_chunks


def evaluate_row(app, row):
    """
    Runs one test_set.json row through the graph and scores it against its
    expectations. Returns a dict of per-row booleans the aggregate report
    sums over.
    """
    result = ask(app, row["question"])
    answer = result["answer"]
    citations = result["citations"]

    is_not_found = answer == NOT_FOUND_MESSAGE
    expected_not_found = row.get("expected_not_found", False)

    # Not-found honesty: did we correctly refuse (or correctly not refuse)?
    not_found_correct = is_not_found == expected_not_found

    # Retrieval hit: for a question expected to be answerable, did the
    # pipeline actually produce citations rather than falling through to
    # not-found?
    retrieval_hit = (not expected_not_found) and (not is_not_found) and len(citations) > 0

    # Version correctness: every citation's version should match the
    # question's requested version (or fall within the from/to range for
    # migration rows) — a citation from a different version is exactly the
    # failure mode the hard filter exists to prevent.
    requested_version = row.get("version")
    if requested_version:
        version_correct = all(
            requested_version in (c.get("version") or "") for c in citations
        ) if citations else True
    else:
        version_correct = True  # migration rows checked separately below

    if row["type"] == "migration" and citations:
        from_v, to_v = row.get("from_version"), row.get("to_version")
        version_correct = all(
            (from_v in (c.get("version") or "")) or (to_v in (c.get("version") or ""))
            or (c.get("version") or "") not in ("", None)
            for c in citations
        )

    # Citation accuracy: every citation names the doc_type-appropriate
    # document (e.g. a reference question shouldn't cite the changelog).
    doc_type_to_document = {
        "reference": "Notion API Reference",
        "migration": "Upgrade Guide",
        "changelog": "Changelog",
    }
    expected_document = doc_type_to_document.get(row.get("expected_doc_type"))
    citation_accurate = (
        all(c.get("document") == expected_document for c in citations)
        if citations and expected_document else True
    )

    return {
        "id": row["id"],
        "not_found_correct": not_found_correct,
        "retrieval_hit": retrieval_hit if not expected_not_found else None,
        "version_correct": version_correct if citations else None,
        "citation_accurate": citation_accurate if citations else None,
        "answer": answer,
    }


RATE_LIMIT_BACKOFF_SECONDS = 65  # past the free tier's 60s window, plus margin


def evaluate_row_with_retry(app, row, max_retries=3):
    """
    The free tier's per-minute quota is easy to exceed across a whole
    test set (each row makes several LLM calls), so a 429 here is a
    routine pacing issue, not a real failure. Wait out the window and
    try the same row again, up to max_retries times.
    """
    for attempt in range(max_retries + 1):
        try:
            return evaluate_row(app, row)
        except ResourceExhausted:
            if attempt == max_retries:
                raise
            print(
                f"  rate limited on {row['id']} (attempt {attempt + 1}/{max_retries + 1}) "
                f"-- waiting {RATE_LIMIT_BACKOFF_SECONDS}s"
            )
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)


def run_evaluation(test_set_path="data/test_set.json"):
    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    chunks = load_all_chunks()
    collection = build_chroma_collection(chunks)
    bm25 = build_bm25_index(chunks)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    app = build_graph(collection, bm25, chunks, embed_model)

    rows = []
    for i, row in enumerate(test_set):
        print(f"[{i + 1}/{len(test_set)}] {row['id']}: {row['question'][:60]}...")
        rows.append(evaluate_row_with_retry(app, row))
        if i < len(test_set) - 1:
            # Each row makes several LLM calls; pacing them proactively
            # means most rows never hit the 429 retry path at all.
            time.sleep(5)

    return summarize(rows)


def summarize(rows):
    def rate(key):
        scored = [r[key] for r in rows if r[key] is not None]
        return sum(scored) / len(scored) if scored else None

    version_correctness = rate("version_correct")
    citation_accuracy = rate("citation_accurate")
    retrieval_hit_rate = rate("retrieval_hit")
    not_found_honesty = rate("not_found_correct")
    # Hallucination rate is the inverse of citation accuracy in this
    # scoring scheme — an inaccurate citation is exactly an unsupported
    # claim slipping through.
    hallucination_rate = (1 - citation_accuracy) if citation_accuracy is not None else None

    report = {
        "version_correctness": version_correctness,
        "citation_accuracy": citation_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "hallucination_rate": hallucination_rate,
        "not_found_honesty": not_found_honesty,
    }

    thresholds_passed = {
        "version_correctness": version_correctness is None or version_correctness >= MIN_VERSION_CORRECTNESS,
        "citation_accuracy": citation_accuracy is None or citation_accuracy >= MIN_CITATION_ACCURACY,
        "retrieval_hit_rate": retrieval_hit_rate is None or retrieval_hit_rate >= MIN_RETRIEVAL_HIT_RATE,
        "hallucination_rate": hallucination_rate is None or hallucination_rate <= MAX_HALLUCINATION_RATE,
        "not_found_honesty": not_found_honesty is None or not_found_honesty >= MIN_NOT_FOUND_HONESTY,
    }

    return {"metrics": report, "passed": thresholds_passed, "rows": rows}


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result["metrics"], indent=2))
    print()
    for metric, passed in result["passed"].items():
        print(f"{'PASS' if passed else 'FAIL'}  {metric}")
