"""
embedding pipeline + Chroma vector store + BM25 sparse index +
hybrid retrieval (hard filter -> dense + BM25 -> reciprocal rank fusion).
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.chroma_config import COLLECTION_NAME, PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME
from src.chunking.migration_chunker import chunk_migration_file  # adjust import to your actual function names
from src.chunking.changelog_chunker import chunk_changelog
from src.chunking.reference_chunker import chunk_reference_file
from src.data_sources import DATA_SOURCES


# ---------------------------------------------------------------------------
# 1. Load + chunk everything 
# ---------------------------------------------------------------------------

def load_all_chunks():
    """
    Runs every chunker over its matching files (per src/data_sources.py)
    and returns one flat list of chunks, each with a 'text' + 'metadata'
    (or equivalent flat dict — adjust field names to match what your
    chunkers actually return).
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
# 2. Embed + store in Chroma
# ---------------------------------------------------------------------------

def build_chroma_collection(chunks):
    """
    Embeds every chunk's text and stores it in a persistent Chroma
    collection, with metadata attached for hard filtering later.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Chroma metadata values must be str/int/float/bool — flatten anything
    # like a 'tags' list into a comma-joined string before storing.
    metadatas = []
    for c in chunks:
        meta = {k: v for k, v in c.items() if k != "text"}
        for key, val in list(meta.items()):
            if isinstance(val, list):
                meta[key] = ", ".join(str(v) for v in val)
            elif val is None:
                # Chroma metadata values must be str/int/float/bool — a
                # bare None (e.g. an unsectioned changelog chunk) errors
                # on add(), so drop the key instead of sending null.
                del meta[key]
        metadatas.append(meta)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return collection


# ---------------------------------------------------------------------------
# 3. BM25 sparse index (in-memory, over the same chunk texts)
# ---------------------------------------------------------------------------

def build_bm25_index(chunks):
    """
    BM25 needs tokenized text, not raw strings. Simple whitespace+lowercase
    tokenization is enough here — no need for a full NLP tokenizer for
    API docs vocabulary.
    """
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


# ---------------------------------------------------------------------------
# 4. Hybrid retrieval: hard filter -> dense + BM25 -> RRF merge
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rank_lists, k=60):
    """
    rank_lists: list of ranked id-lists, e.g. [dense_ranked_ids, bm25_ranked_ids]
    Each id's fused score = sum over lists of 1 / (k + rank_in_that_list).
    k=60 is the standard RRF default from the original paper - dampens the
    impact of any single list's top result dominating the fusion.
    """
    scores = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


def hybrid_retrieve(query, collection, bm25, chunks, model, doc_type=None, version=None, top_k=3):
    """
    Hard filter FIRST (per PRD: filtering, not ranking, is what prevents
    wrong-version answers), then run dense + BM25 over the filtered set,
    then merge via RRF.

    top_k=3 rather than 5: grading calls the LLM once per retrieved chunk,
    so this directly sets how many grading calls one question makes. On
    the free tier's per-minute quota, that's the gap between one question
    fitting under the limit and not.
    """
    # --- Hard filter: which chunk indices survive doc_type/version filter ---
    def version_matches(chunk):
        if version is None:
            return True
        # Migration chunks carry a single "version" string. Reference
        # chunks carry a "versions" list (the OpenAPI spec covers more
        # than one API version at once). Changelog release_date isn't an
        # API version and is intentionally not matched here.
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

    # --- Dense retrieval (over full collection, then filtered to allowed ids) ---
    query_embedding = model.encode([query]).tolist()
    dense_results = collection.query(query_embeddings=query_embedding, n_results=len(chunks))
    dense_ranked = [doc_id for doc_id in dense_results["ids"][0] if doc_id in filtered_ids]

    # --- BM25 retrieval (over full corpus, then filtered) ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_all = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranked = [f"chunk_{i}" for i in bm25_ranked_all if f"chunk_{i}" in filtered_ids]

    # --- Fuse ---
    fused_ids = reciprocal_rank_fusion([dense_ranked, bm25_ranked])[:top_k]
    fused_indices = [int(doc_id.split("_")[1]) for doc_id in fused_ids]

    return [chunks[i] for i in fused_indices]