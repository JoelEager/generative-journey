from typing import Optional
import anthropic

from generative_journey.ai_clients import BaseAIClient


class AnthropicClient(BaseAIClient):
    """AI client using Anthropic API SDK."""

    provider_name: str = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(model=model)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        results = []
        for block in response.content:
            if hasattr(block, "text"):
                results.append(block.text)

        return "".join(results)
