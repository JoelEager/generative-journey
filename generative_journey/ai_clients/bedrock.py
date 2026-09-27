"""AI client using AWS Bedrock runtime (boto3) Converse API."""
from typing import List, Any, Tuple

from .common import BaseAIClient
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, parse_tool_action


class BedrockClient(BaseAIClient):
    provider_name: str = "bedrock"

    def __init__(self, model="amazon.nova-pro-v1:0"):
        super().__init__(model=model)

        # Lazy import to avoid unnecessary dependency if this client is not used
        import boto3
        self.client = boto3.client("bedrock-runtime")

    def format_user_message(self, content: Any) -> dict:
        if isinstance(content, str):
            return {"role": "user", "content": [{"text": content}]}
        return {"role": "user", "content": content}

    def _format_tools(self, tools: List[dict]) -> Any:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": {"json": tool["parameters"]},
                    }
                }
                for tool in tools
            ]
        }

    def _call_model(self, formatted_tools: Any) -> Any:
        kwargs = {
            "modelId": self.model,
            "inferenceConfig": {"maxTokens": 1024},
            "system": [{"text": ai_prompt.SYSTEM_PROMPT}],
            "toolConfig": formatted_tools,
            "messages": self.messages,
        }
        return self.client.converse(**kwargs)

    def _parse_response(self, response: Any) -> Tuple[List[Any], List[Any], bool, List[Any], List[Any]]:
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        turn_actions = []
        verbose_actions = []
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
                    action = parse_tool_action(fn_name, args)
                    turn_actions.append(action)
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
                    verbose_actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": f"Error: {error_msg}. Please try again with valid parameters."}],
                            "status": "error",
                        }
                    })
            else:
                verbose_actions.append(VerboseMessage(repr(block), type="unknown"))

        assistant_msgs = [{"role": "assistant", "content": content_blocks}]
        retry_msgs = [{"role": "user", "content": tool_results}] if tool_results else []

        return turn_actions, verbose_actions, has_invalid_tool, assistant_msgs, retry_msgs
