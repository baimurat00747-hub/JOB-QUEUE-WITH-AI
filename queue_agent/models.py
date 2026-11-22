import os
from dataclasses import dataclass

from fastapi import HTTPException
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    url: str
    api_key: str | None = None

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self):
        if self.provider == "ollama":
            return ChatOllama(
                model=self.model, base_url=self.url.rstrip("/").removesuffix("/v1"),
                temperature=0, client_kwargs={"timeout": 90},
            )
        return ChatOpenAI(
            model=self.model, base_url=self.url, api_key=self.api_key,
            timeout=90, max_retries=0,
        )


def model_settings():
    provider = os.environ.get("QUEUE_AGENT_PROVIDER", "ollama")
    if provider == "ollama":
        return ModelSettings(
            provider, os.environ.get("QUEUE_AGENT_MODEL", "qwen3:4b"),
            os.environ.get("QUEUE_AGENT_MODEL_URL", "http://127.0.0.1:11434/v1"),
        )
    if provider == "openai":
        model = os.environ.get("QUEUE_AGENT_MODEL")
        key = os.environ.get("QUEUE_AGENT_API_KEY")
        if not model or not key:
            raise HTTPException(503, "Configure QUEUE_AGENT_MODEL and QUEUE_AGENT_API_KEY.")
        return ModelSettings(provider, model, os.environ.get("QUEUE_AGENT_MODEL_URL", "https://api.openai.com/v1"), key)
    raise HTTPException(503, "Unknown queue model provider.")
