"""Shared implementation for AI clients."""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple

MAX_INVOCATIONS = 3


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
        from .. import ai_prompt
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
    def _parse_response(self, response: Any) -> Tuple[List[Any], List[Any], bool, List[Any], List[Any]]:
        """Parse provider response into actions and history entries.

        Returns a tuple of:
          - turn_actions: List of valid action instances (NarrativeMessage, EndGame, etc.)
          - verbose_actions: List of VerboseMessage objects (e.g. tool call warnings)
          - has_invalid_tool: bool indicating if any tool invocation was invalid
          - assistant_msgs: List of messages representing the assistant turn to append to history
          - retry_msgs: List of messages representing tool results/errors to append to history
        """
        pass

    def generate_actions(self) -> List[Any]:
        """Generates a list of action objects (NarrativeMessage, VerboseMessage, EndGame) for the turn.

        Reads current prompt state from generative_journey.ai_prompt and updates client message history.

        :return: List of action instances.
        """
        from .. import ai_prompt

        self.append_user_prompt()
        formatted_tools = self._format_tools(ai_prompt.TOOLS)
        actions: List[Any] = []

        for _ in range(MAX_INVOCATIONS):
            response = self._call_model(formatted_tools)
            turn_actions, verbose_actions, has_invalid_tool, assistant_msgs, retry_msgs = self._parse_response(response)

            actions.extend(verbose_actions)
            self.messages.extend(assistant_msgs)
            self.messages.extend(retry_msgs)

            if has_invalid_tool:
                continue

            actions.extend(turn_actions)
            break

        return actions
