import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key():
    return os.environ["OPENROUTER_API_KEY"]


chat_model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
    api_key=_api_key(),
    base_url=OPENROUTER_BASE_URL,
    max_tokens=2048,
)

embedding_function = OpenAIEmbeddings(
    model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    api_key=_api_key(),
    base_url=OPENROUTER_BASE_URL,
    tiktoken_enabled=False,
)


def generate_text(prompt: str) -> str:
    response = chat_model.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)
