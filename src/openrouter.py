import os

import requests
from langchain_openai import ChatOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return key


def _get_chat_model():
    """Lazy initialisation — avoids crashing at import time if the env var
    isn't set yet (e.g. during the Render build phase)."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
        api_key=_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

class OpenRouterEmbeddings:
    def __init__(self):
        self.model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.url = f"{OPENROUTER_BASE_URL}/embeddings"

    def embed_documents(self, texts):
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {_api_key()}"},
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    def embed_query(self, text):
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
