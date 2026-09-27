"""AI client using AWS Bedrock runtime (boto3) Converse API."""
from typing import List, Any

from .common import BaseAIClient, MAX_INVOCATIONS
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, EndGame


class BedrockClient(BaseAIClient):
    provider_name: str = "bedrock"

    def __init__(self, model="amazon.nova-pro-v1:0"):
        super().__init__(model=model)

        # Lazy import to avoid unnecessary dependency if this client is not used
        import boto3
        self.client = boto3.client("bedrock-runtime")
        self.messages: List[dict] = []

    def generate_actions(self) -> List[Any]:
        self.messages.append({"role": "user", "content": [{"text": ai_prompt.current_prompt}]})

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

        for _ in range(MAX_INVOCATIONS):
            kwargs = {
                "modelId": self.model,
                "inferenceConfig": {"maxTokens": 1024},
                "system": [{"text": ai_prompt.SYSTEM_PROMPT}],
                "toolConfig": tool_config,
                "messages": self.messages,
            }

            response = self.client.converse(**kwargs)
            content_blocks = response.get("output", {}).get("message", {}).get("content", [])

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
                else:
                    actions.append(VerboseMessage(repr(block), type="unknown"))

            # Record assistant turn in message history
            self.messages.append({"role": "assistant", "content": content_blocks})

            if has_invalid_tool:
                # Append tool result errors to message history as user message for retry
                self.messages.append({"role": "user", "content": tool_results})
                continue

            if tool_results:
                # Add successful tool results to message history as user message for next turn
                self.messages.append({"role": "user", "content": tool_results})

            actions.extend(turn_actions)
            break  # Exit the loop after a successful turn without invalid tools
        
        return actions
