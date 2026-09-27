"""Shared implementation for AI clients."""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple

from .. import ai_prompt

MAX_INVOCATIONS = 3
MAX_TOKENS = 1024


class BaseAIClient(ABC):
    """Abstract base client interface for AI storytelling interaction."""

    provider_name: str = "base"

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.messages: List[Any] = []

    def __repr__(self) -> str:
        return f"provider: {self.provider_name}, model: {self.model}"

    def format_user_message(self, content: Any) -> dict:
        """Formats a user message object for conversation history."""
        return {"role": "user", "content": content}

    def append_user_prompt(self) -> None:
        """Appends the current user prompt from generative_journey.ai_prompt to message history."""
        self.messages.append(self.format_user_message(ai_prompt.current_prompt))

    @abstractmethod
    def _format_tools(self, tools: List[dict]) -> Any:
        """Format generic prompt tools for provider-specific API calls."""
        pass

    @abstractmethod
    def _call_model(self, formatted_tools: Any) -> Any:
        """Invoke provider-specific API call with current messages and formatted tools."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> Tuple[List[Any], bool]:
        """
        Parse provider response into actions and update messages history.

        Returns a tuple of:
          - actions: List of action instances from this invocation (NarrativeMessage, EndGame, etc.)
          - has_invalid_tool: bool indicating if any tool invocation was invalid
        """
        pass

    def generate_actions(self) -> List[Any]:
        """Generate a list of action objects (NarrativeMessage, EndGame, etc.) for the turn."""
        self.append_user_prompt()
        formatted_tools = self._format_tools(ai_prompt.TOOLS)
        actions: List[Any] = []

        for _ in range(MAX_INVOCATIONS):
            response = self._call_model(formatted_tools)
            resp_actions, has_invalid_tool = self._parse_response(response)
            actions.extend(resp_actions)
            if not has_invalid_tool:
                break

        return actions
