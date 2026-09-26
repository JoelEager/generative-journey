from typing import Optional
import openai

from generative_journey.ai_clients import BaseAIClient


class OpenAIClient(BaseAIClient):
    """AI client using OpenAI API SDK."""

    provider_name: str = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model=model)
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

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
