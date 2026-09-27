"""AI client using OpenAI API SDK."""
import json
from typing import List, Any, Tuple
from os import getenv

from .common import BaseAIClient
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, parse_tool_action


class OpenAIClient(BaseAIClient):
    provider_name: str = "openai"

    def __init__(self, model="gpt-4o", api_key=None, base_url=None):
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
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "system": ai_prompt.SYSTEM_PROMPT,
            "messages": self.messages,
            "tools": formatted_tools,
        }
        return self.client.chat.completions.create(**kwargs)

    def _parse_response(self, response: Any) -> Tuple[List[Any], List[Any], bool, List[Any], List[Any]]:
        turn_actions = []
        verbose_actions = []
        has_invalid_tool = False
        assistant_msgs = []
        retry_msgs = []

        for entry in response.choices:
            message = entry.message
            assistant_msgs.append(message)

            if message.content:
                turn_actions.append(NarrativeMessage(message.content))

            if message.tool_calls:
                tool_errors = []
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        action = parse_tool_action(fn_name, args)
                        turn_actions.append(action)
                    except Exception as err:
                        has_invalid_tool = True
                        error_msg = str(err)
                        tool_errors.append((tool_call, error_msg))
                        verbose_actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={raw_args}: {error_msg}", type="warn"))

                if has_invalid_tool:
                    for tool_call, error_msg in tool_errors:
                        retry_msgs.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}. Please try again with valid parameters."
                        })

        return turn_actions, verbose_actions, has_invalid_tool, assistant_msgs, retry_msgs
