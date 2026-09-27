"""AI client using OpenAI API SDK."""
import json
from typing import List, Any
from os import getenv

from .common import BaseAIClient, MAX_INVOCATIONS
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, EndGame


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
        self.messages: List[dict] = []

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        if self.base_url:
            repr_str += f", url: {self.base_url}"
        return repr_str

    def generate_actions(self) -> List[Any]:
        self.messages.append({"role": "user", "content": ai_prompt.current_prompt})

        tools = [
            {
                "type": "function",
                "function": tool,
            }
            for tool in ai_prompt.TOOLS
        ]

        actions: List[Any] = []

        for _ in range(MAX_INVOCATIONS):
            kwargs = {
                "model": self.model,
                "max_tokens": 1024,
                "system": ai_prompt.SYSTEM_PROMPT,
                "messages": self.messages,
                "tools": tools,
            }

            response = self.client.chat.completions.create(**kwargs)

            turn_actions = []
            has_invalid_tool = False
            tool_errors = []

            for entry in response.choices:
                message = entry.message

                if message.content:
                    turn_actions.append(NarrativeMessage(message.content))

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        fn_name = tool_call.function.name
                        raw_args = tool_call.function.arguments
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            if not isinstance(args, dict) or "won" not in args or not isinstance(args["won"], bool):
                                raise ValueError(f"Invalid arguments for {fn_name}: {raw_args}")
                            if fn_name == "end_game":
                                turn_actions.append(EndGame(won=args["won"]))
                            else:
                                raise ValueError(f"Unknown tool name: {fn_name}")
                        except Exception as err:
                            has_invalid_tool = True
                            error_msg = str(err)
                            tool_errors.append((tool_call, error_msg))
                            actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={raw_args}: {error_msg}", type="warn"))

                if has_invalid_tool:
                    # Add assistant message and tool error responses to conversation history so AI can retry
                    self.messages.append(message)
                    for tool_call, error_msg in tool_errors:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}. Please try again with valid parameters."
                        })
                    continue

                # Successful invocation
                self.messages.append(message)
            
            actions.extend(turn_actions)
        
        return actions
