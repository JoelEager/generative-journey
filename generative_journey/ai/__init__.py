from abc import ABC, abstractmethod
from typing import Dict, Optional


DEFAULT_MODELS: Dict[str, str] = {
    "bedrock": "amazon.nova-pro-v1:0",
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "local": "local-model",
}


class BaseAIClient(ABC):
    """Abstract base client interface for AI storytelling interaction."""

    def __init__(self, model: str):
        self.model = model

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
    if provider_clean not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported AI provider: '{provider}'. Supported providers are: {', '.join(DEFAULT_MODELS.keys())}"
        )

    selected_model = model if model else DEFAULT_MODELS[provider_clean]

    if provider_clean == "bedrock":
        from generative_journey.ai.bedrock import BedrockClient
        return BedrockClient(model=selected_model)
    elif provider_clean == "anthropic":
        from generative_journey.ai.anthropic import AnthropicClient
        return AnthropicClient(model=selected_model)
    elif provider_clean == "openai":
        from generative_journey.ai.openai import OpenAIClient
        return OpenAIClient(model=selected_model)
    elif provider_clean == "local":
        from generative_journey.ai.local import LocalClient
        return LocalClient(model=selected_model)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")
