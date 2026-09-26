"""Shared implementation for AI clients."""
from abc import ABC, abstractmethod
from typing import List, Optional, Any

MAX_INVOCATIONS = 3


class BaseAIClient(ABC):
    """Abstract base client interface for AI storytelling interaction."""

    provider_name: str = "base"

    def __init__(self, model: Optional[str] = None):
        self.model = model

    def __repr__(self) -> str:
        return f"provider: {self.provider_name}, model: {self.model}"

    @abstractmethod
    def generate_actions(self) -> List[Any]:
        """Generates a list of action objects (NarrativeMessage, VerboseMessage, EndGame) for the turn.

        Reads current prompt state from generative_journey.ai_prompt and updates client message history.

        :return: List of action instances.
        """
        pass
