"""AI client using Anthropic API."""
import json
from typing import Optional, List, Any
from os import getenv

from generative_journey.ai_clients.common import BaseAIClient, MAX_INVOCATIONS
from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame


class AnthropicClient(BaseAIClient):
    provider_name: str = "anthropic"

    def __init__(self, model="claude-3-5-sonnet-20241022", api_key=None):
        super().__init__(model=model)
        if not api_key:
            api_key = getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        
        # Lazy import to avoid unnecessary dependency if this client is not used
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages: List[dict] = []

    def generate_actions(self) -> List[Any]:
        if not self.messages:
            user_text = ai_prompt.PLAYER_MESSAGE if ai_prompt.PLAYER_MESSAGE else "Start the game."
            self.messages.append({"role": "user", "content": user_text})
        else:
            if ai_prompt.PLAYER_MESSAGE:
                self.messages.append({"role": "user", "content": ai_prompt.PLAYER_MESSAGE})

        tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in ai_prompt.TOOLS
        ]

        actions: List[Any] = []

        for invocation in range(MAX_INVOCATIONS):
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

            # Record assistant turn in message history
            self.messages.append({"role": "assistant", "content": response.content})

            if has_invalid_tool:
                # Append tool result errors to message history as user message for retry
                self.messages.append({"role": "user", "content": tool_results})
                if invocation == MAX_INVOCATIONS - 1:
                    raise RuntimeError("AI failed to invoke tool with valid arguments within maximum allowed attempts.")
                continue

            if tool_results:
                # Add successful tool results if needed
                self.messages.append({"role": "user", "content": tool_results})

            actions.extend(turn_actions)
            return actions

        raise RuntimeError("AI failed to generate actions within maximum allowed attempts.")
