import os
from typing import Optional
import openai

from generative_journey.ai_clients import BaseAIClient


class LocalClient(BaseAIClient):
    """AI client for OpenAI-compatible local AI servers (e.g., LM Studio, Ollama, vLLM, LocalAI)."""

    provider_name: str = "local"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model=model)
        self.base_url = (
            base_url
            or os.environ.get("LOCAL_AI_BASE_URL", "http://localhost:1234/v1")
        )
        api_key = os.environ.get("LOCAL_AI_API_KEY", "local-ai")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)

    def __repr__(self) -> str:
        return f"provider: local, url: {self.base_url}"

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {"messages": messages}
        if self.model:
            kwargs["model"] = self.model

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""
