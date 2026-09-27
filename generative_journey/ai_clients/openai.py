"""AI client using OpenAI API SDK."""
import json
from typing import List, Any, Tuple
from os import getenv

from .common import BaseAIClient
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, parse_tool_action


class OpenAIClient(BaseAIClient):
    provider_name: str = "openai"

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        super().__init__(model=model)
        self.base_url = base_url

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        else:
            api_key = getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            kwargs["api_key"] = api_key

        if base_url:
            kwargs["base_url"] = base_url
        else:
            base_url = getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url

        # Lazy import to avoid unnecessary dependency if this client is not used
        import openai
        self.client = openai.OpenAI(**kwargs)

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        if self.base_url:
            repr_str += f", url: {self.base_url}"
        return repr_str

    def _format_tools(self, tools: List[dict]) -> Any:
        return [
            {
                "type": "function",
                "function": tool,
            }
            for tool in tools
        ]

    def _call_model(self, formatted_tools: Any) -> Any:
        messages = [{"role": "system", "content": ai_prompt.SYSTEM_PROMPT}] + self.messages
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": formatted_tools,
        }
        return self.client.chat.completions.create(**kwargs)

    def _format_tool_result(self, tool_call_id: str, content: str = "Success") -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def _parse_response(self, response: Any) -> Tuple[List[Any], bool]:
        actions = [
            VerboseMessage(f"{response.model} {response.usage}", type="usage")
        ]
        has_invalid_tool = False

        for entry in response.choices:
            message = entry.message
            self.messages.append(message)

            if getattr(message, "reasoning_content", None):
                actions.append(VerboseMessage(message.reasoning_content, type="thinking"))

            if getattr(message, "content", None):
                actions.append(NarrativeMessage(message.content))

            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    func = tool_call.function
                    actions.append(VerboseMessage(repr(func), type="toolUse"))
                    try:
                        args = json.loads(func.arguments)
                        action = parse_tool_action(func.name, args)
                        actions.append(action)
                        self.messages.append(self._format_tool_result(tool_call.id, content="Success"))
                    except Exception as err:
                        has_invalid_tool = True
                        error_msg = str(err)
                        actions.append(VerboseMessage(f"Failed tool call '{func.name}' args={func.arguments}: {error_msg}", type="warn"))
                        self.messages.append(self._format_tool_result(
                            tool_call.id,
                            content=f"Error: {error_msg}. Please try again with valid parameters."
                        ))

        return actions, has_invalid_tool
