import os
import time

import requests
from langchain_openai import ChatOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Module-level cache — built once on first use, reused for every subsequent
# generate_text call. Avoids creating a new ChatOpenAI object (which
# validates credentials and builds an HTTP client) on every LLM call.
_chat_model_cache: ChatOpenAI | None = None


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return key


def _get_chat_model() -> ChatOpenAI:
    """Returns a cached ChatOpenAI instance, creating it on first call."""
    global _chat_model_cache
    if _chat_model_cache is None:
        _chat_model_cache = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
            api_key=_api_key(),
            base_url=OPENROUTER_BASE_URL,
        )
    return _chat_model_cache


class OpenRouterEmbeddings:
    def __init__(self):
        self.model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.url = f"{OPENROUTER_BASE_URL}/embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Retry up to 5 times with exponential backoff on 429 rate-limit
        # responses. The index build at startup embeds all chunks in batches
        # and a single transient 429 should never abort the whole startup.
        for attempt in range(5):
            response = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {_api_key()}"},
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            if response.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()["data"]
            return [
                item["embedding"]
                for item in sorted(data, key=lambda item: item["index"])
            ]
        # All retries exhausted — raise the last 429 as a real error.
        response.raise_for_status()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


embedding_function = OpenRouterEmbeddings()


def generate_text(prompt: str, max_tokens: int = 256) -> str:
    response = _get_chat_model().bind(max_tokens=max_tokens).invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)
