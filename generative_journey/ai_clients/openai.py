"""AI client using OpenAI API SDK."""
import json
from typing import List, Any, Tuple, Optional
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

    def _format_tool_result(self, tool_call_id: str, content: str = "Success") -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def _parse_response(self, response: Any) -> Tuple[List[Any], bool]:
        def _get_val(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _get_reasoning_content(msg) -> Optional[str]:
            if isinstance(msg, dict):
                return msg.get("reasoning_content") or msg.get("thinking")
            rc = getattr(msg, "reasoning_content", None) or getattr(msg, "thinking", None)
            if rc:
                return rc
            model_extra = getattr(msg, "model_extra", None)
            if isinstance(model_extra, dict):
                return model_extra.get("reasoning_content") or model_extra.get("thinking")
            return None

        actions = []
        has_invalid_tool = False
        choices = _get_val(response, "choices") or []

        for entry in choices:
            message = _get_val(entry, "message")
            if not message:
                continue

            if isinstance(message, dict):
                assistant_msg = {k: v for k, v in message.items() if k not in ("reasoning_content", "thinking")}
            else:
                assistant_msg = message
            self.messages.append(assistant_msg)

            thinking_text = _get_reasoning_content(message)
            if thinking_text:
                actions.append(VerboseMessage(thinking_text, type="thinking"))

            content = _get_val(message, "content")
            if content:
                actions.append(NarrativeMessage(content))

            tool_calls = _get_val(message, "tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    actions.append(VerboseMessage(repr(tool_call), type="toolUse"))
                    func = _get_val(tool_call, "function")
                    fn_name = _get_val(func, "name") if func else None
                    raw_args = _get_val(func, "arguments") if func else {}
                    tool_id = _get_val(tool_call, "id")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        action = parse_tool_action(fn_name, args)
                        actions.append(action)
                        self.messages.append(self._format_tool_result(tool_id, content="Success"))
                    except Exception as err:
                        has_invalid_tool = True
                        error_msg = str(err)
                        actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={raw_args}: {error_msg}", type="warn"))
                        self.messages.append(self._format_tool_result(
                            tool_id,
                            content=f"Error: {error_msg}. Please try again with valid parameters."
                        ))

        return actions, has_invalid_tool
