import os
from typing import Optional
import openai

from generative_journey.ai import BaseAIClient


class LocalClient(BaseAIClient):
    """AI client for OpenAI-compatible local AI servers (e.g., LM Studio, Ollama, vLLM, LocalAI)."""

    def __init__(self, model: str, base_url: Optional[str] = None):
        super().__init__(model)
        url = (
            base_url
            or os.environ.get("LOCAL_AI_BASE_URL")
            or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        )
        api_key = os.environ.get("LOCAL_AI_API_KEY") or os.environ.get("LMSTUDIO_API_KEY", "local-ai")
        self.client = openai.OpenAI(base_url=url, api_key=api_key)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""
