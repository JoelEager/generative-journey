from unittest.mock import MagicMock, patch
import pytest
from click.testing import CliRunner

from generative_journey.ai_clients import BaseAIClient, SUPPORTED_PROVIDERS, get_ai_client
from generative_journey.ai_clients.anthropic import AnthropicClient
from generative_journey.ai_clients.bedrock import BedrockClient
from generative_journey.ai_clients.local import LocalClient
from generative_journey.ai_clients.openai import OpenAIClient
from generative_journey.cli import main


@patch("generative_journey.ai_clients.openai.openai.OpenAI")
@patch("generative_journey.ai_clients.anthropic.anthropic.Anthropic")
@patch("generative_journey.ai_clients.bedrock.boto3.client")
def test_get_ai_client_factory(mock_boto, mock_anthropic, mock_openai):
    bedrock = get_ai_client("bedrock")
    assert isinstance(bedrock, BedrockClient)
    assert bedrock.model == "amazon.nova-pro-v1:0"
    assert repr(bedrock) == "provider: bedrock, model: amazon.nova-pro-v1:0"

    anthropic = get_ai_client("anthropic", model="custom-claude")
    assert isinstance(anthropic, AnthropicClient)
    assert anthropic.model == "custom-claude"
    assert repr(anthropic) == "provider: anthropic, model: custom-claude"

    openai = get_ai_client("openai")
    assert isinstance(openai, OpenAIClient)
    assert openai.model == "gpt-4o"
    assert repr(openai) == "provider: openai, model: gpt-4o"

    local = get_ai_client("local")
    assert isinstance(local, LocalClient)
    assert local.model is None
    assert repr(local) == "provider: local, url: http://localhost:1234/v1"

    with pytest.raises(ValueError, match="Unsupported AI provider"):
        get_ai_client("unknown_provider")


@patch("generative_journey.ai_clients.bedrock.boto3.client")
def test_bedrock_client_generate_response(mock_boto_client):
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

    assert response == "Welcome to the enchanted forest!"
    mock_bedrock.converse.assert_called_once_with(
        modelId="amazon.nova-pro-v1:0",
        messages=[{"role": "user", "content": [{"text": "Start adventure"}]}],
        system=[{"text": "You are a narrator."}],
    )


@patch("generative_journey.ai_clients.anthropic.anthropic.Anthropic")
def test_anthropic_client_generate_response(mock_anthropic_cls):
    mock_anthropic_instance = MagicMock()
    mock_anthropic_cls.return_value = mock_anthropic_instance
    mock_block = MagicMock()
    mock_block.text = "You stand before a glowing cavern."
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_anthropic_instance.messages.create.return_value = mock_response

    client = AnthropicClient(model="claude-3-5-sonnet-20241022", api_key="dummy_key")
    response = client.generate_response("Look around")

    assert response == "You stand before a glowing cavern."
    mock_anthropic_instance.messages.create.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Look around"}],
    )


@patch("generative_journey.ai_clients.openai.openai.OpenAI")
def test_openai_client_generate_response(mock_openai_cls):
    mock_openai_instance = MagicMock()
    mock_openai_cls.return_value = mock_openai_instance
    mock_choice = MagicMock()
    mock_choice.message.content = "A quiet tavern sits on the hill."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_instance.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4o", api_key="dummy_key")
    response = client.generate_response("Describe location", system_prompt="Dungeon master mode")

    assert response == "A quiet tavern sits on the hill."
    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Dungeon master mode"},
            {"role": "user", "content": "Describe location"},
        ],
    )


@patch("generative_journey.ai_clients.local.openai.OpenAI")
def test_local_client_generate_response_without_model(mock_openai_cls):
    mock_openai_instance = MagicMock()
    mock_openai_cls.return_value = mock_openai_instance
    mock_choice = MagicMock()
    mock_choice.message.content = "You are in a dimly lit dungeon chamber."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_instance.chat.completions.create.return_value = mock_response

    client = LocalClient(base_url="http://localhost:1234/v1")
    response = client.generate_response("Look at surroundings", system_prompt="RPG game host")

    assert response == "You are in a dimly lit dungeon chamber."
    mock_openai_cls.assert_called_once_with(base_url="http://localhost:1234/v1", api_key="local-ai")
    mock_openai_instance.chat.completions.create.assert_called_once_with(
        messages=[
            {"role": "system", "content": "RPG game host"},
            {"role": "user", "content": "Look at surroundings"},
        ],
    )


@patch("generative_journey.ai_clients.local.openai.OpenAI")
def test_local_client_generate_response_with_model(mock_openai_cls):
    mock_openai_instance = MagicMock()
    mock_openai_cls.return_value = mock_openai_instance
    mock_choice = MagicMock()
    mock_choice.message.content = "You are in a dimly lit dungeon chamber."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_instance.chat.completions.create.return_value = mock_response

    client = LocalClient(model="llama-3-8b", base_url="http://localhost:1234/v1")
    response = client.generate_response("Look at surroundings")

    assert response == "You are in a dimly lit dungeon chamber."
    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model="llama-3-8b",
        messages=[
            {"role": "user", "content": "Look at surroundings"},
        ],
    )


def test_cli_missing_provider_exits_with_error():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 1
    assert "Error: Missing required argument 'PROVIDER'." in result.output
    assert "Please refer to README.md" in result.output


def test_cli_unsupported_provider_exits_with_error():
    runner = CliRunner()
    result = runner.invoke(main, ["invalid"])
    assert result.exit_code == 1
    assert "Error: Unsupported AI provider 'invalid'." in result.output
    assert "Please refer to README.md" in result.output


@patch("generative_journey.cli.get_ai_client")
def test_cli_successful_invocation(mock_get_ai_client):
    mock_client = MagicMock()
    mock_client.__str__.return_value = "provider: local, url: http://localhost:1234/v1"
    mock_repr = MagicMock()
    mock_client.__repr__ = MagicMock(return_value="provider: local, url: http://localhost:1234/v1")
    mock_client.generate_response.return_value = "The journey begins under a moonlit sky."
    mock_get_ai_client.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["local", "-m", "llama-3-8b"])

    assert result.exit_code == 0
    assert "Welcome to Generative Journey! (provider: local, url: http://localhost:1234/v1)" in result.output
    assert "AI Response:\nThe journey begins under a moonlit sky." in result.output
    mock_get_ai_client.assert_called_once_with("local", model="llama-3-8b")
