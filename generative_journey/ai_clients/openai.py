"""AI client using OpenAI API SDK."""
import json
from typing import Optional, List, Any
from os import getenv

from generative_journey.ai_clients.common import BaseAIClient, MAX_INVOCATIONS
from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame


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
        if not self.messages:
            self.messages.append({"role": "system", "content": ai_prompt.SYSTEM_PROMPT})
            user_text = ai_prompt.PLAYER_MESSAGE if ai_prompt.PLAYER_MESSAGE else "Start the game."
            self.messages.append({"role": "user", "content": user_text})
        else:
            if ai_prompt.PLAYER_MESSAGE:
                self.messages.append({"role": "user", "content": ai_prompt.PLAYER_MESSAGE})

        tools = [
            {
                "type": "function",
                "function": tool,
            }
            for tool in ai_prompt.TOOLS
        ]

        actions: List[Any] = []

        for invocation in range(MAX_INVOCATIONS):
            kwargs = {
                "messages": self.messages,
                "tools": tools,
            }
            if self.model and self.model != "default":
                kwargs["model"] = self.model

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            turn_actions = []
            has_invalid_tool = False
            tool_errors = []

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
                if invocation == MAX_INVOCATIONS - 1:
                    raise RuntimeError("AI failed to invoke tool with valid arguments within maximum allowed attempts.")
                continue

            # Successful invocation
            self.messages.append(message)
            actions.extend(turn_actions)
            return actions

        raise RuntimeError("AI failed to generate actions within maximum allowed attempts.")
