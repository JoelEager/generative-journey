"""AI client interface and factory for Generative Journey."""
from .common import BaseAIClient
from .bedrock import BedrockClient
from .openai import OpenAIClient
from .local import LocalClient

CLIENTS = (BedrockClient, OpenAIClient, LocalClient)


def get_ai_client(provider: str, **kwargs) -> BaseAIClient:
    """
    Factory function to instantiate an AI client based on provider name.
    :param provider: Name of provider ('bedrock', 'openai', 'local').
    :param **kwargs: Additional keyword arguments for the AI client.
    :return: An instance of BaseAIClient.
    :raises ValueError: If provider is unknown.
    """
    for client_class in CLIENTS:
        if client_class.provider_name == provider:
            return client_class(**kwargs)

    raise ValueError(
        f"Unsupported AI provider: '{provider}'. Supported providers are: {', '.join([client.provider_name for client in CLIENTS])}"
    )
