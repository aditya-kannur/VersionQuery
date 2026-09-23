"""
LLM + embeddings provider module.

Previously backed by OpenRouter/OpenAI. Now uses:
  - Groq (chat completions) - free tier, fast inference
  - sentence-transformers (embeddings) — runs locally, no API key needed

Public interface is unchanged: generate_text() and embedding_function.
All other modules import from here and need no changes.
"""
import os
from typing import Any


# ---------------------------------------------------------------------------
# Groq chat models
# ---------------------------------------------------------------------------

DEFAULT_GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
DEFAULT_MAX_TOKENS = 384

_chat_model_cache: dict[str, Any] = {}


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return key


def _configured_models() -> list[str]:
    """
    Returns Groq model IDs to try in order.

    GROQ_MODEL can force a preferred model, and GROQ_FALLBACK_MODELS can add
    comma-separated backups. 70B goes first: generation quality (obeying
    "ground strictly in this text, never reproduce JSON") matters more than
    speed, and the 8B model was observed leaking memorized real-world JSON
    examples from its own pretraining on well-known public API docs, even
    when that JSON was never in the retrieved context. 8B stays as a
    same-request fallback for accounts where 70B access is unavailable.
    """
    configured = []
    preferred = os.getenv("GROQ_MODEL")
    if preferred:
        configured.append(preferred)

    fallback_env = os.getenv("GROQ_FALLBACK_MODELS")
    if fallback_env:
        configured.extend(model.strip() for model in fallback_env.split(","))
    elif not configured:
        configured.extend(DEFAULT_GROQ_MODELS)
    else:
        configured.extend(DEFAULT_GROQ_MODELS)

    models = []
    for model in configured:
        if model and model not in models:
            models.append(model)
    return models


def _get_chat_model(model_name: str) -> Any:
    """Returns a cached ChatGroq instance for model_name."""
    if model_name not in _chat_model_cache:
        from langchain_groq import ChatGroq

        _chat_model_cache[model_name] = ChatGroq(
            model=model_name,
            api_key=_api_key(),
            timeout=30,
            max_retries=1,
        )
    return _chat_model_cache[model_name]


def _token_budget(requested_max_tokens: int) -> int:
    configured_limit = int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    return max(1, min(requested_max_tokens, configured_limit))


def generate_text(prompt: str, max_tokens: int = 256) -> str:
    last_error: Exception | None = None
    for model_name in _configured_models():
        try:
            response = (
                _get_chat_model(model_name)
                .bind(max_tokens=_token_budget(max_tokens))
                .invoke(prompt)
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        tried = ", ".join(_configured_models())
        raise RuntimeError(f"Groq request failed for configured models: {tried}") from last_error

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

_embedding_model_cache: Any | None = None


def _get_embedding_model() -> Any:
    """Returns a cached SentenceTransformer instance, loaded on first call."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embedding_model_cache = SentenceTransformer(model_name)
    return _embedding_model_cache


class LocalEmbeddings:
    """
    Drop-in replacement for the old OpenRouterEmbeddings class.
    Same embed_documents / embed_query interface, backed by
    sentence-transformers running locally - no API key for inference.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


embedding_function = LocalEmbeddings()