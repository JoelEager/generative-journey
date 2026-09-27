import unittest
from unittest.mock import MagicMock, patch

from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame
from generative_journey.ai_clients import BaseAIClient, get_ai_client
from generative_journey.ai_clients.anthropic import AnthropicClient
from generative_journey.ai_clients.bedrock import BedrockClient
from generative_journey.ai_clients.local import LocalClient
from generative_journey.ai_clients.openai import OpenAIClient


class DummyAIClient(BaseAIClient):
    provider_name = "dummy"

    def _format_tools(self, tools):
        return tools

    def _call_model(self, formatted_tools):
        return None

    def _parse_response(self, response):
        return [], False


class TestBaseAIClient(unittest.TestCase):

    def test_base_client_init(self):
        client = DummyAIClient(model="dummy-model")
        self.assertEqual(client.model, "dummy-model")
        self.assertEqual(client.messages, [])
        self.assertEqual(str(client), "provider: dummy, model: dummy-model")

    def test_format_user_message(self):
        client = DummyAIClient()
        formatted = client.format_user_message("Hello world")
        self.assertEqual(formatted, {"role": "user", "content": "Hello world"})

    def test_append_user_prompt(self):
        client = DummyAIClient()
        ai_prompt.current_prompt = "Test prompt content"
        client.append_user_prompt()
        self.assertEqual(len(client.messages), 1)
        self.assertEqual(client.messages[0], {"role": "user", "content": "Test prompt content"})


class TestAIClientsFactory(unittest.TestCase):

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy_anthropic_key"})
    @patch("openai.OpenAI")
    @patch("anthropic.Anthropic")
    @patch("boto3.client")
    def test_get_ai_client_factory(self, mock_boto, mock_anthropic, mock_openai):
        bedrock = get_ai_client("bedrock")
        self.assertIsInstance(bedrock, BedrockClient)
        self.assertEqual(bedrock.model, "amazon.nova-pro-v1:0")

        anthropic = get_ai_client("anthropic", model="custom-claude")
        self.assertIsInstance(anthropic, AnthropicClient)
        self.assertEqual(anthropic.model, "custom-claude")

        openai = get_ai_client("openai", api_key="dummy")
        self.assertIsInstance(openai, OpenAIClient)
        self.assertEqual(openai.model, "gpt-4o")

        local = get_ai_client("local")
        self.assertIsInstance(local, LocalClient)

        with self.assertRaises(ValueError):
            get_ai_client("unknown_provider")


class TestOpenAIClient(unittest.TestCase):

    @patch("openai.OpenAI")
    def test_openai_generate_actions_success(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        mock_msg = MagicMock()
        mock_msg.content = "You awaken in a dark room."
        mock_msg.tool_calls = None
        mock_msg.reasoning_content = None
        mock_msg.thinking = None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        ai_prompt.current_prompt = "Look around"
        actions = client.generate_actions()

        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], NarrativeMessage)
        self.assertEqual(actions[0].message, "You awaken in a dark room.")

    @patch("openai.OpenAI")
    def test_openai_generate_actions_tool_call(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "end_game"
        mock_tool_call.function.arguments = '{"won": true}'

        mock_msg = MagicMock()
        mock_msg.content = "Victory is yours!"
        mock_msg.tool_calls = [mock_tool_call]
        mock_msg.reasoning_content = None
        mock_msg.thinking = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertEqual(len(actions), 3) # toolUse VerboseMessage, NarrativeMessage, EndGame
        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "toolUse" for a in actions))
        self.assertTrue(any(isinstance(a, NarrativeMessage) and a.message == "Victory is yours!" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and a.won for a in actions))

    @patch("openai.OpenAI")
    def test_openai_generate_actions_invalid_tool_retry(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        # 1st response: invalid arguments
        mock_bad_tool = MagicMock()
        mock_bad_tool.id = "call_1"
        mock_bad_tool.function.name = "end_game"
        mock_bad_tool.function.arguments = '{"invalid_param": 123}'
        mock_bad_msg = MagicMock(content="Attempting end game", tool_calls=[mock_bad_tool], reasoning_content=None, thinking=None)
        resp1 = MagicMock(choices=[MagicMock(message=mock_bad_msg)])

        # 2nd response: fixed tool call
        mock_good_tool = MagicMock()
        mock_good_tool.id = "call_2"
        mock_good_tool.function.name = "end_game"
        mock_good_tool.function.arguments = '{"won": false}'
        mock_good_msg = MagicMock(content="Game lost", tool_calls=[mock_good_tool], reasoning_content=None, thinking=None)
        resp2 = MagicMock(choices=[MagicMock(message=mock_good_msg)])

        mock_instance.chat.completions.create.side_effect = [resp1, resp2]

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "warn" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and not a.won for a in actions))

    @patch("openai.OpenAI")
    def test_openai_thinking_trace_and_history_exclusion(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        dict_message = {
            "role": "assistant",
            "content": "You see a chest.",
            "reasoning_content": "Pondering options: chest might be trapped.",
        }
        mock_response = MagicMock(choices=[MagicMock(message=dict_message)])
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        # Check that VerboseMessage was generated with reasoning text
        thinking_msgs = [a for a in actions if isinstance(a, VerboseMessage) and a.type == "thinking"]
        self.assertEqual(len(thinking_msgs), 1)
        self.assertEqual(thinking_msgs[0].message, "Pondering options: chest might be trapped.")

        # Check history exclusion: self.messages should NOT contain "reasoning_content"
        assistant_hist = [m for m in client.messages if isinstance(m, dict) and m.get("role") == "assistant"]
        self.assertEqual(len(assistant_hist), 1)
        self.assertNotIn("reasoning_content", assistant_hist[0])
        self.assertEqual(assistant_hist[0]["content"], "You see a chest.")


class TestAnthropicClient(unittest.TestCase):

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy_anthropic_key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_generate_actions(self, mock_anthropic_cls):
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "You see a dark portal."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_1"
        mock_tool_block.name = "end_game"
        mock_tool_block.input = {"won": True}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_instance.messages.create.return_value = mock_response

        client = AnthropicClient()
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, NarrativeMessage) and a.message == "You see a dark portal." for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and a.won for a in actions))

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy_anthropic_key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_generate_actions_invalid_tool_retry(self, mock_anthropic_cls):
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        mock_bad_tool = MagicMock()
        mock_bad_tool.type = "tool_use"
        mock_bad_tool.id = "tool_1"
        mock_bad_tool.name = "end_game"
        mock_bad_tool.input = {"invalid_param": 123}
        resp1 = MagicMock(content=[mock_bad_tool])

        mock_good_tool = MagicMock()
        mock_good_tool.type = "tool_use"
        mock_good_tool.id = "tool_2"
        mock_good_tool.name = "end_game"
        mock_good_tool.input = {"won": False}
        resp2 = MagicMock(content=[mock_good_tool])

        mock_instance.messages.create.side_effect = [resp1, resp2]

        client = AnthropicClient()
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "warn" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and not a.won for a in actions))

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy_anthropic_key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_thinking_trace_and_history_exclusion(self, mock_anthropic_cls):
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        mock_thinking = MagicMock()
        mock_thinking.type = "thinking"
        mock_thinking.thinking = "Analyzing user intent carefully."

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "The room is quiet."

        mock_response = MagicMock(content=[mock_thinking, mock_text])
        mock_instance.messages.create.return_value = mock_response

        client = AnthropicClient()
        actions = client.generate_actions()

        thinking_msgs = [a for a in actions if isinstance(a, VerboseMessage) and a.type == "thinking"]
        self.assertEqual(len(thinking_msgs), 1)
        self.assertEqual(thinking_msgs[0].message, "Analyzing user intent carefully.")

        # Ensure thinking block was excluded from self.messages history
        assistant_hist = [m for m in client.messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_hist), 1)
        hist_content = assistant_hist[0]["content"]
        self.assertEqual(len(hist_content), 1)
        self.assertEqual(hist_content[0], mock_text)


class TestBedrockClient(unittest.TestCase):

    @patch("boto3.client")
    def test_bedrock_generate_actions(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_boto.return_value = mock_bedrock

        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "A mystical beacon shines."},
                        {"toolUse": {"toolUseId": "tu_1", "name": "end_game", "input": {"won": True}}}
                    ]
                }
            }
        }
        mock_bedrock.converse.return_value = mock_response

        client = BedrockClient()
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, NarrativeMessage) and a.message == "A mystical beacon shines." for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and a.won for a in actions))

    @patch("boto3.client")
    def test_bedrock_generate_actions_invalid_tool_retry(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_boto.return_value = mock_bedrock

        resp1 = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "tu_1", "name": "end_game", "input": {"bad_arg": 1}}}
                    ]
                }
            }
        }

        resp2 = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "tu_2", "name": "end_game", "input": {"won": False}}}
                    ]
                }
            }
        }
        mock_bedrock.converse.side_effect = [resp1, resp2]

        client = BedrockClient()
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "warn" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and not a.won for a in actions))

    @patch("boto3.client")
    def test_bedrock_thinking_trace_and_history_exclusion(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_boto.return_value = mock_bedrock

        thinking_block = {"reasoningContent": {"reasoningText": {"text": "Evaluating tactical situation."}}}
        text_block = {"text": "You see an exit."}

        mock_response = {
            "output": {
                "message": {
                    "content": [thinking_block, text_block]
                }
            }
        }
        mock_bedrock.converse.return_value = mock_response

        client = BedrockClient()
        actions = client.generate_actions()

        thinking_msgs = [a for a in actions if isinstance(a, VerboseMessage) and a.type == "thinking"]
        self.assertEqual(len(thinking_msgs), 1)
        self.assertEqual(thinking_msgs[0].message, "Evaluating tactical situation.")

        # Check self.messages excludes reasoningContent block
        assistant_hist = [m for m in client.messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_hist), 1)
        self.assertEqual(assistant_hist[0]["content"], [text_block])

    @patch("boto3.client")
    def test_bedrock_nova_thinking_tags_and_history_stripping(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_boto.return_value = mock_bedrock

        nova_text_block = {
            "text": "<thinking>\nNeed to describe the ancient ruins.\n</thinking>\nYou stand before ancient, ivy-covered ruins."
        }

        mock_response = {
            "output": {
                "message": {
                    "content": [nova_text_block]
                }
            }
        }
        mock_bedrock.converse.return_value = mock_response

        client = BedrockClient()
        actions = client.generate_actions()

        # Verify thinking trace action
        thinking_msgs = [a for a in actions if isinstance(a, VerboseMessage) and a.type == "thinking"]
        self.assertEqual(len(thinking_msgs), 1)
        self.assertEqual(thinking_msgs[0].message, "Need to describe the ancient ruins.")

        # Verify narrative message
        narrative_msgs = [a for a in actions if isinstance(a, NarrativeMessage)]
        self.assertEqual(len(narrative_msgs), 1)
        self.assertEqual(narrative_msgs[0].message, "You stand before ancient, ivy-covered ruins.")

        # Verify history stripped <thinking> tag
        assistant_hist = [m for m in client.messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_hist), 1)
        self.assertEqual(assistant_hist[0]["content"], [{"text": "You stand before ancient, ivy-covered ruins."}])


if __name__ == "__main__":
    unittest.main()
