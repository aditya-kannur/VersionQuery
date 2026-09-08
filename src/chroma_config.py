"""
Static Chroma vector store configuration.
No logic — just constants used by the embedding/retrieval code later.
"""

COLLECTION_NAME = "versionquery_chunks"
PERSIST_DIRECTORY = "data/chroma_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
