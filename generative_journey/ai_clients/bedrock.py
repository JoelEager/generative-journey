"""AI client using AWS Bedrock runtime (boto3) Converse API."""
from typing import Optional, List, Any

from generative_journey.ai_clients.common import BaseAIClient, MAX_INVOCATIONS
from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame


class BedrockClient(BaseAIClient):
    provider_name: str = "bedrock"

    def __init__(self, model="amazon.nova-pro-v1:0"):
        super().__init__(model=model)

        # Lazy import to avoid unnecessary dependency if this client is not used
        import boto3
        self.client = boto3.client("bedrock-runtime")
        self.messages: List[dict] = []

    def generate_actions(self) -> List[Any]:
        if not self.messages:
            user_text = ai_prompt.PLAYER_MESSAGE if ai_prompt.PLAYER_MESSAGE else "Start the game."
            self.messages.append({
                "role": "user",
                "content": [{"text": user_text}]
            })
        else:
            if ai_prompt.PLAYER_MESSAGE:
                self.messages.append({
                    "role": "user",
                    "content": [{"text": ai_prompt.PLAYER_MESSAGE}]
                })

        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": {"json": tool["parameters"]},
                    }
                }
                for tool in ai_prompt.TOOLS
            ]
        }

        actions: List[Any] = []

        for invocation in range(MAX_INVOCATIONS):
            kwargs = {
                "modelId": self.model,
                "messages": self.messages,
                "system": [{"text": ai_prompt.SYSTEM_PROMPT}],
                "toolConfig": tool_config,
            }

            response = self.client.converse(**kwargs)
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            turn_actions = []
            has_invalid_tool = False
            tool_results = []

            for block in content_blocks:
                if "text" in block:
                    if block["text"]:
                        turn_actions.append(NarrativeMessage(block["text"]))
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_use_id = tool_use.get("toolUseId")
                    fn_name = tool_use.get("name")
                    args = tool_use.get("input", {})
                    try:
                        if not isinstance(args, dict) or "won" not in args or not isinstance(args["won"], bool):
                            raise ValueError(f"Invalid arguments for {fn_name}: {args}")
                        if fn_name == "end_game":
                            turn_actions.append(EndGame(won=args["won"]))
                        else:
                            raise ValueError(f"Unknown tool name: {fn_name}")
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": "Success"}],
                                "status": "success",
                            }
                        })
                    except Exception as err:
                        has_invalid_tool = True
                        error_msg = str(err)
                        actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": f"Error: {error_msg}. Please try again with valid parameters."}],
                                "status": "error",
                            }
                        })

            # Record assistant turn in message history
            self.messages.append({
                "role": "assistant",
                "content": content_blocks,
            })

            if has_invalid_tool:
                # Append tool result errors to message history as user message for retry
                self.messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                if invocation == MAX_INVOCATIONS - 1:
                    raise RuntimeError("AI failed to invoke tool with valid arguments within maximum allowed attempts.")
                continue

            if tool_results:
                self.messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            actions.extend(turn_actions)
            return actions

        raise RuntimeError("AI failed to generate actions within maximum allowed attempts.")
