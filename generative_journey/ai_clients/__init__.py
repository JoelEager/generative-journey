from abc import ABC, abstractmethod
from typing import Dict, Optional, Type


SUPPORTED_PROVIDERS = ("bedrock", "anthropic", "openai", "local")


class BaseAIClient(ABC):
    """Abstract base client interface for AI storytelling interaction."""

    provider_name: str = "base"

    def __init__(self, model: Optional[str] = None):
        self.model = model

    def __repr__(self) -> str:
        return f"provider: {self.provider_name}, model: {self.model}"

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a text response from the underlying AI model.

        :param prompt: User prompt input.
        :param system_prompt: Optional system prompt or persona instructions.
        :return: Generated string response.
        """
        pass


def get_ai_client(provider: str, model: Optional[str] = None) -> BaseAIClient:
    """Factory function to instantiate an AI client based on provider name.

    :param provider: Name of provider ('bedrock', 'anthropic', 'openai', 'local').
    :param model: Optional model override.
    :return: An instance of BaseAIClient.
    :raises ValueError: If provider is unknown.
    """
    provider_clean = provider.lower().strip()
    if provider_clean not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported AI provider: '{provider}'. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    kwargs = {}
    if model is not None:
        kwargs["model"] = model

    if provider_clean == "bedrock":
        from generative_journey.ai_clients.bedrock import BedrockClient
        return BedrockClient(**kwargs)
    elif provider_clean == "anthropic":
        from generative_journey.ai_clients.anthropic import AnthropicClient
        return AnthropicClient(**kwargs)
    elif provider_clean == "openai":
        from generative_journey.ai_clients.openai import OpenAIClient
        return OpenAIClient(**kwargs)
    elif provider_clean == "local":
        from generative_journey.ai_clients.local import LocalClient
        return LocalClient(**kwargs)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")
