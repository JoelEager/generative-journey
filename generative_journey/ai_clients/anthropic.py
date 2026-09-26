"""AI client using Anthropic API."""
from typing import Optional
from os import getenv

from . import BaseAIClient


class AnthropicClient(BaseAIClient):
    provider_name: str = "anthropic"

    def __init__(self, model="claude-3-5-sonnet-20241022"):
        super().__init__(model=model)
        api_key = getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        
        # Lazy import to avoid unnecessary dependency if this client is not used
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

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
