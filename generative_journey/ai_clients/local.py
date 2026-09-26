"""AI client for local LLM runtimes that expose an OpenAI-compatible API."""
from .openai import OpenAIClient

class LocalClient(OpenAIClient):
    provider_name: str = "local"

    def __init__(self, **_):
        super().__init__(model="default", api_key="placeholder", base_url="http://localhost:1234/v1")
