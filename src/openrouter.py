"""
LLM + embeddings provider module.

Previously backed by OpenRouter/OpenAI. Now uses:
  - Groq (chat completions) — free tier, fast inference
  - sentence-transformers (embeddings) — runs locally, no API key needed

Public interface is unchanged: generate_text() and embedding_function.
All other modules import from here and need no changes.
"""
import os

from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Groq chat model
# ---------------------------------------------------------------------------

_chat_model_cache: ChatGroq | None = None


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return key


def _get_chat_model() -> ChatGroq:
    """Returns a cached ChatGroq instance, created on first call."""
    global _chat_model_cache
    if _chat_model_cache is None:
        _chat_model_cache = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=_api_key(),
        )
    return _chat_model_cache


def generate_text(prompt: str, max_tokens: int = 256) -> str:
    response = _get_chat_model().bind(max_tokens=max_tokens).invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Sentence-transformers embeddings (runs in-process, no API key needed)
# ---------------------------------------------------------------------------

_embedding_model_cache: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Returns a cached SentenceTransformer instance, loaded on first call."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embedding_model_cache = SentenceTransformer(model_name)
    return _embedding_model_cache


class LocalEmbeddings:
    """
    Drop-in replacement for the old OpenRouterEmbeddings class.
    Same embed_documents / embed_query interface, backed by
    sentence-transformers running locally — no network call, no API key.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


embedding_function = LocalEmbeddings()
