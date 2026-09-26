import unittest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from generative_journey.ai_clients import BaseAIClient, SUPPORTED_PROVIDERS, get_ai_client
from generative_journey.ai_clients.anthropic import AnthropicClient
from generative_journey.ai_clients.bedrock import BedrockClient
from generative_journey.ai_clients.local import LocalClient
from generative_journey.ai_clients.openai import OpenAIClient
from generative_journey.cli import main


class TestAIClientsAndCLI(unittest.TestCase):

    @patch("generative_journey.ai_clients.openai.openai.OpenAI")
    @patch("generative_journey.ai_clients.anthropic.anthropic.Anthropic")
    @patch("generative_journey.ai_clients.bedrock.boto3.client")
    def test_get_ai_client_factory(self, mock_boto, mock_anthropic, mock_openai):
        bedrock = get_ai_client("bedrock")
        self.assertIsInstance(bedrock, BedrockClient)
        self.assertEqual(bedrock.model, "amazon.nova-pro-v1:0")
        self.assertEqual(repr(bedrock), "provider: bedrock, model: amazon.nova-pro-v1:0")

        anthropic = get_ai_client("anthropic", model="custom-claude")
        self.assertIsInstance(anthropic, AnthropicClient)
        self.assertEqual(anthropic.model, "custom-claude")
        self.assertEqual(repr(anthropic), "provider: anthropic, model: custom-claude")

        openai = get_ai_client("openai")
        self.assertIsInstance(openai, OpenAIClient)
        self.assertEqual(openai.model, "gpt-4o")
        self.assertEqual(repr(openai), "provider: openai, model: gpt-4o")

        local = get_ai_client("local")
        self.assertIsInstance(local, LocalClient)
        self.assertIsNone(local.model)
        self.assertEqual(repr(local), "provider: local, url: http://localhost:1234/v1")

        with self.assertRaises(ValueError) as ctx:
            get_ai_client("unknown_provider")
        self.assertIn("Unsupported AI provider", str(ctx.exception))

    @patch("generative_journey.ai_clients.bedrock.boto3.client")
    def test_bedrock_client_generate_response(self, mock_boto_client):
        mock_bedrock = MagicMock()
        mock_boto_client.return_value = mock_bedrock
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Welcome to the enchanted forest!"}]
                }
            }
        }

        client = BedrockClient(model="amazon.nova-pro-v1:0")
        response = client.generate_response("Start adventure", system_prompt="You are a narrator.")

        self.assertEqual(response, "Welcome to the enchanted forest!")
        mock_bedrock.converse.assert_called_once_with(
            modelId="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"text": "Start adventure"}]}],
            system=[{"text": "You are a narrator."}],
        )

    @patch("generative_journey.ai_clients.anthropic.anthropic.Anthropic")
    def test_anthropic_client_generate_response(self, mock_anthropic_cls):
        mock_anthropic_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_anthropic_instance
        mock_block = MagicMock()
        mock_block.text = "You stand before a glowing cavern."
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_anthropic_instance.messages.create.return_value = mock_response

        client = AnthropicClient(model="claude-3-5-sonnet-20241022", api_key="dummy_key")
        response = client.generate_response("Look around")

        self.assertEqual(response, "You stand before a glowing cavern.")
        mock_anthropic_instance.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Look around"}],
        )

    @patch("generative_journey.ai_clients.openai.openai.OpenAI")
    def test_openai_client_generate_response(self, mock_openai_cls):
        mock_openai_instance = MagicMock()
        mock_openai_cls.return_value = mock_openai_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "A quiet tavern sits on the hill."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4o", api_key="dummy_key")
        response = client.generate_response("Describe location", system_prompt="Dungeon master mode")

        self.assertEqual(response, "A quiet tavern sits on the hill.")
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Dungeon master mode"},
                {"role": "user", "content": "Describe location"},
            ],
        )

    @patch("generative_journey.ai_clients.local.openai.OpenAI")
    def test_local_client_generate_response_without_model(self, mock_openai_cls):
        mock_openai_instance = MagicMock()
        mock_openai_cls.return_value = mock_openai_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "You are in a dimly lit dungeon chamber."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_instance.chat.completions.create.return_value = mock_response

        client = LocalClient(base_url="http://localhost:1234/v1")
        response = client.generate_response("Look at surroundings", system_prompt="RPG game host")

        self.assertEqual(response, "You are in a dimly lit dungeon chamber.")
        mock_openai_cls.assert_called_once_with(base_url="http://localhost:1234/v1", api_key="local-ai")
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": "RPG game host"},
                {"role": "user", "content": "Look at surroundings"},
            ],
        )

    @patch("generative_journey.ai_clients.local.openai.OpenAI")
    def test_local_client_generate_response_with_model(self, mock_openai_cls):
        mock_openai_instance = MagicMock()
        mock_openai_cls.return_value = mock_openai_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "You are in a dimly lit dungeon chamber."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_instance.chat.completions.create.return_value = mock_response

        client = LocalClient(model="llama-3-8b", base_url="http://localhost:1234/v1")
        response = client.generate_response("Look at surroundings")

        self.assertEqual(response, "You are in a dimly lit dungeon chamber.")
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model="llama-3-8b",
            messages=[
                {"role": "user", "content": "Look at surroundings"},
            ],
        )

    def test_cli_missing_provider_exits_with_error(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Missing required argument 'PROVIDER'.", result.output)
        self.assertIn("Please refer to README.md", result.output)

    def test_cli_unsupported_provider_exits_with_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["invalid"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Unsupported AI provider 'invalid'.", result.output)
        self.assertIn("Please refer to README.md", result.output)

    @patch("generative_journey.cli.get_ai_client")
    def test_cli_successful_invocation(self, mock_get_ai_client):
        mock_client = MagicMock()
        mock_client.__repr__ = MagicMock(return_value="provider: local, url: http://localhost:1234/v1")
        mock_client.generate_response.return_value = "The journey begins under a moonlit sky."
        mock_get_ai_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["local", "-m", "llama-3-8b"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Welcome to Generative Journey! (provider: local, url: http://localhost:1234/v1)", result.output)
        self.assertIn("AI Response:\nThe journey begins under a moonlit sky.", result.output)
        mock_get_ai_client.assert_called_once_with("local", model="llama-3-8b")


if __name__ == "__main__":
    unittest.main()
