"""AI client using Anthropic API."""
from typing import List, Any, Tuple, Optional
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

    def _format_tool_result(self, tool_use_id: str, is_error: bool = False, text: Optional[str] = None) -> dict:
        result = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text if text else ("Error" if is_error else "Success"),
        }
        if is_error:
            result["is_error"] = True
        return result

    def _parse_response(self, response: Any) -> Tuple[List[Any], bool]:
        def _get_val(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        content_blocks = getattr(response, "content", []) or []
        history_blocks = [b for b in content_blocks if _get_val(b, "type") != "thinking"]
        if history_blocks:
            self.messages.append({"role": "assistant", "content": history_blocks})

        actions = []
        tool_results = []
        has_invalid_tool = False

        for block in content_blocks:
            block_type = _get_val(block, "type")
            if block_type == "thinking":
                thinking_text = _get_val(block, "thinking") or _get_val(block, "text") or str(block)
                actions.append(VerboseMessage(thinking_text, type="thinking"))
            elif block_type == "text":
                text = _get_val(block, "text")
                if text:
                    actions.append(NarrativeMessage(text))
            elif block_type == "tool_use":
                actions.append(VerboseMessage(repr(block), type="toolUse"))
                fn_name = _get_val(block, "name")
                args = _get_val(block, "input") or {}
                tool_id = _get_val(block, "id")
                try:
                    action = parse_tool_action(fn_name, args)
                    actions.append(action)
                    tool_results.append(self._format_tool_result(tool_id, is_error=False, text="Success"))
                except Exception as err:
                    has_invalid_tool = True
                    error_msg = str(err)
                    actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                    tool_results.append(self._format_tool_result(
                        tool_id,
                        is_error=True,
                        text=f"Error: {error_msg}. Please try again with valid parameters."
                    ))
            else:
                actions.append(VerboseMessage(repr(block), type="unknown"))

        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})

        return actions, has_invalid_tool
