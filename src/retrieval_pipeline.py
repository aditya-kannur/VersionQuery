"""
Embedding pipeline + in-memory numpy vector store + BM25 sparse index +
hybrid retrieval (hard filter -> dense + BM25 -> reciprocal rank fusion).

ChromaDB was replaced with a lightweight numpy-based vector store to avoid
the ~1.2 GB hnswlib/chromadb build dependency. Functionality is identical:
cosine similarity search over pre-computed embeddings, with the same hard
metadata filtering and RRF fusion as before.
"""
import time
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking.migration_chunker import chunk_migration_file
from src.chunking.changelog_chunker import chunk_changelog
from src.chunking.reference_chunker import chunk_reference_file
from src.data_sources import DATA_SOURCES


# ---------------------------------------------------------------------------
# 1. Load + chunk everything
# ---------------------------------------------------------------------------

def load_all_chunks():
    """
    Runs every chunker over its matching files (per src/data_sources.py)
    and returns one flat list of chunks, each a dict with at minimum a
    'text' key plus doc-type-specific metadata fields.
    """
    all_chunks = []

    for source in DATA_SOURCES:
        doc_type = source["doc_type"]
        path = source["path"]

        if doc_type == "migration":
            chunks = chunk_migration_file(path)
        elif doc_type == "changelog":
            chunks = chunk_changelog(open(path, encoding="utf-8").read(), doc_type="changelog")
        elif doc_type == "reference":
            chunks = chunk_reference_file(path)
        else:
            continue

        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# 2. Numpy vector store (replaces ChromaDB)
# ---------------------------------------------------------------------------

class NumpyVectorStore:
    """
    Pure-numpy cosine similarity store. Holds embeddings as a float32
    matrix so similarity search is a single batched dot product — no
    external libraries, no disk writes, no build step.

    Interface is intentionally minimal: build once at startup, query many
    times per request, same as the ChromaDB collection it replaces.
    """

    def __init__(self, embeddings: list[list[float]], ids: list[str]):
        matrix = np.array(embeddings, dtype=np.float32)
        # L2-normalise rows so dot product == cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero on zero vectors
        self._matrix = matrix / norms
        self._ids = ids

    def query(self, query_embedding: list[float], n_results: int) -> list[str]:
        """
        Returns up to n_results chunk IDs sorted by descending cosine
        similarity to query_embedding.
        """
        vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores = self._matrix @ vec          # shape (n_chunks,)
        top_n = min(n_results, len(self._ids))
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._ids[i] for i in top_indices]


def build_chroma_collection(chunks, embedding_function):
    """
    Kept with the original name so api.py and graph.py need zero changes.
    Builds and returns a NumpyVectorStore instead of a Chroma collection.
    """
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    all_embeddings = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        for attempt in range(5):
            try:
                batch = embedding_function.embed_documents(texts[start:start + batch_size])
                all_embeddings.extend(batch)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    return NumpyVectorStore(all_embeddings, ids)


# ---------------------------------------------------------------------------
# 3. BM25 sparse index (unchanged)
# ---------------------------------------------------------------------------

def build_bm25_index(chunks):
    """
    BM25 needs tokenized text, not raw strings. Simple whitespace+lowercase
    tokenization is enough here — no need for a full NLP tokenizer for
    API docs vocabulary.
    """
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized_corpus)


# ---------------------------------------------------------------------------
# 4. Hybrid retrieval: hard filter -> dense + BM25 -> RRF merge (unchanged)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rank_lists, k=60):
    """
    rank_lists: list of ranked id-lists, e.g. [dense_ranked_ids, bm25_ranked_ids]
    Each id's fused score = sum over lists of 1 / (k + rank_in_that_list).
    k=60 is the standard RRF default — dampens the impact of any single
    list's top result dominating the fusion.
    """
    scores = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


def hybrid_retrieve(query, collection, bm25, chunks, embedding_function,
                    doc_type=None, version=None, top_k=3):
    """
    Hard filter FIRST (filtering, not ranking, is what prevents wrong-version
    answers), then dense + BM25 over the filtered set, then RRF merge.

    `collection` is now a NumpyVectorStore but the call signature is
    identical to the previous ChromaDB version — nothing upstream changes.
    """
    # --- Hard filter ---
    def version_matches(chunk):
        if version is None:
            return True
        if "versions" in chunk:
            return version in chunk["versions"]
        return chunk.get("version") == version

    filtered_indices = [
        i for i, c in enumerate(chunks)
        if (doc_type is None or c.get("doc_type") == doc_type)
        and version_matches(c)
    ]
    if not filtered_indices:
        return []

    filtered_ids = {f"chunk_{i}" for i in filtered_indices}

    # --- Dense retrieval ---
    query_embedding = embedding_function.embed_query(query)
    dense_ranked = [
        doc_id for doc_id in collection.query(query_embedding, n_results=len(chunks))
        if doc_id in filtered_ids
    ]

    # --- BM25 retrieval ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = [
        f"chunk_{i}"
        for i in sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
        if f"chunk_{i}" in filtered_ids
    ]

    # --- Fuse ---
    fused_ids = reciprocal_rank_fusion([dense_ranked, bm25_ranked])[:top_k]
    fused_indices = [int(doc_id.split("_")[1]) for doc_id in fused_ids]
    return [chunks[i] for i in fused_indices]
