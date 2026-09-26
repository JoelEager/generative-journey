"""Shared implementation for AI clients."""
from abc import ABC, abstractmethod
from typing import Optional

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
