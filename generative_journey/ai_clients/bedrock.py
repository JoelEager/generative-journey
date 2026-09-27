"""AI client using AWS Bedrock runtime (boto3) Converse API."""
import re
from typing import List, Any, Optional

from .common import BaseAIClient
from .. import ai_prompt
from ..ai_actions import NarrativeMessage, VerboseMessage, parse_tool_action

THINKING_PATTERN = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)


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

    def _format_tools(self, tools: List[dict]):
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

    def _call_model(self, formatted_tools):
        return self.client.converse(
            modelId=self.model,
            inferenceConfig={"maxTokens": 1024},
            system=[{"text": ai_prompt.SYSTEM_PROMPT}],
            toolConfig=formatted_tools,
            messages=self.messages,
        )

    def _format_tool_result(self, tool_use_id: str, status: str, text: Optional[str] = None) -> dict:
        return {
            "toolResult": {
                "toolUseId": tool_use_id,
                "status": status,
                "content": [{"text": text if text else status}],
            }
        }

    def _parse_response(self, response):
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        history_blocks = []
        tool_results = []
        actions = []
        has_invalid_tool = False

        for block in content_blocks:
            if block.get("text"):
                raw_text = block["text"]
                for match in THINKING_PATTERN.finditer(raw_text):
                    thinking_str = match.group(1).strip()
                    if thinking_str:
                        actions.append(VerboseMessage(thinking_str, type="thinking"))
                clean_text = THINKING_PATTERN.sub("", raw_text).strip()
                if clean_text:
                    actions.append(NarrativeMessage(clean_text))
                    history_blocks.append({"text": clean_text})
            elif "toolUse" in block:
                actions.append(VerboseMessage(repr(block), type="toolUse"))
                history_blocks.append(block)
                tool_use = block["toolUse"]
                tool_use_id = tool_use.get("toolUseId")
                fn_name = tool_use.get("name")
                args = tool_use.get("input", {})
                try:
                    actions.append(parse_tool_action(fn_name, args))
                    tool_results.append(self._format_tool_result(tool_use_id, "success"))
                except Exception as err:
                    has_invalid_tool = True
                    error_msg = str(err)
                    actions.append(VerboseMessage(f"Failed tool call '{fn_name}' args={args}: {error_msg}", type="warn"))
                    tool_results.append(self._format_tool_result(tool_use_id, "error", f"Error: {error_msg}. Please try again with valid parameters."))
            else:
                actions.append(VerboseMessage(repr(block), type="unknown"))
                history_blocks.append(block)

        if history_blocks:
            self.messages.append({"role": "assistant", "content": history_blocks})

        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})

        return actions, has_invalid_tool
