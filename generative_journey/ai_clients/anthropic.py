"""AI client using Anthropic API."""
from typing import List, Any, Tuple
from os import getenv

from .common import BaseAIClient
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, parse_tool_action


class AnthropicClient(BaseAIClient):
    provider_name: str = "anthropic"

    def __init__(self, model="claude-3-5-sonnet-20241022"):
        super().__init__(model=model)
        api_key = getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        # Lazy import to avoid unnecessary dependency if this client is not used
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def _format_tools(self, tools: List[dict]) -> Any:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in tools
        ]

    def _call_model(self, formatted_tools: Any) -> Any:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "system": ai_prompt.SYSTEM_PROMPT,
            "tools": formatted_tools,
            "messages": self.messages,
        }
        return self.client.messages.create(**kwargs)

    def _parse_response(self, response: Any) -> Tuple[List[Any], List[Any], bool, List[Any], List[Any]]:
        turn_actions = []
        verbose_actions = []
        has_invalid_tool = False
        tool_results = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                if block.text:
                    turn_actions.append(NarrativeMessage(block.text))
            elif getattr(block, "type", None) == "tool_use":
                fn_name = block.name
                args = block.input
                try:
                    action = parse_tool_action(fn_name, args)
                    turn_actions.append(action)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Success",
                    })
                except Exception as err:
                    has_invalid_tool = True
                    error_msg = str(err)
                    verbose_actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {error_msg}. Please try again with valid parameters.",
                        "is_error": True,
                    })
            else:
                verbose_actions.append(VerboseMessage(repr(block), type="unknown"))

        assistant_msgs = [{"role": "assistant", "content": response.content}]
        retry_msgs = [{"role": "user", "content": tool_results}] if tool_results else []

        return turn_actions, verbose_actions, has_invalid_tool, assistant_msgs, retry_msgs
