"""AI client using AWS Bedrock runtime (boto3) Converse API."""
from typing import Optional

from generative_journey.ai_clients import BaseAIClient


class BedrockClient(BaseAIClient):
    provider_name: str = "bedrock"

    def __init__(self, model="amazon.nova-pro-v1:0"):
        super().__init__(model=model)

        # Lazy import to avoid unnecessary dependency if this client is not used
        import boto3
        self.client = boto3.client("bedrock-runtime")

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
