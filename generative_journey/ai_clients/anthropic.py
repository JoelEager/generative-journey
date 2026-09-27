"""AI client using Anthropic API."""
from typing import List, Any
from os import getenv

from .common import BaseAIClient, MAX_INVOCATIONS
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, EndGame


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
        self.messages: List[dict] = []

    def generate_actions(self) -> List[Any]:
        self.messages.append({"role": "user", "content": ai_prompt.current_prompt})

        tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in ai_prompt.TOOLS
        ]

        actions: List[Any] = []

        for _ in range(MAX_INVOCATIONS):
            kwargs = {
                "model": self.model,
                "max_tokens": 1024,
                "system": ai_prompt.SYSTEM_PROMPT,
                "tools": tools,
                "messages": self.messages,
            }

            response = self.client.messages.create(**kwargs)

            turn_actions = []
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
                        if not isinstance(args, dict) or "won" not in args or not isinstance(args["won"], bool):
                            raise ValueError(f"Invalid arguments for {fn_name}: {args}")
                        if fn_name == "end_game":
                            turn_actions.append(EndGame(won=args["won"]))
                        else:
                            raise ValueError(f"Unknown tool name: {fn_name}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Success",
                        })
                    except Exception as err:
                        has_invalid_tool = True
                        error_msg = str(err)
                        actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {error_msg}. Please try again with valid parameters.",
                            "is_error": True,
                        })
                else:
                    actions.append(VerboseMessage(repr(block), type="unknown"))

            # Record assistant turn in message history
            self.messages.append({"role": "assistant", "content": response.content})

            if has_invalid_tool:
                # Append tool result errors to message history as user message for retry
                self.messages.append({"role": "user", "content": tool_results})
                continue

            if tool_results:
                # Add successful tool results to message history as user message for next turn
                self.messages.append({"role": "user", "content": tool_results})

            actions.extend(turn_actions)
        
        return actions
