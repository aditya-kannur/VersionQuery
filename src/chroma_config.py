"""
Static Chroma vector store configuration.
No logic — just constants used by the embedding/retrieval code later.
"""

COLLECTION_NAME = "versionquery_chunks_openrouter"
PERSIST_DIRECTORY = "data/chroma_store"
EMBEDDING_MODEL_NAME = "openai/text-embedding-3-small"
