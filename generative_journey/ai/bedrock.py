import json
from typing import Optional
import boto3

from generative_journey.ai import BaseAIClient


class BedrockClient(BaseAIClient):
    """AI client using AWS Bedrock runtime (boto3) Converse API."""

    def __init__(self, model: str, region_name: Optional[str] = None):
        super().__init__(model)
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        kwargs = {
            "modelId": self.model,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**kwargs)

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        results = []
        for block in content_blocks:
            if "text" in block:
                results.append(block["text"])

        return "".join(results)
