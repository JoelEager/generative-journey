import unittest
from unittest.mock import MagicMock, patch

from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame
from generative_journey.ai_clients import BaseAIClient, get_ai_client
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

    @patch("openai.OpenAI")
    @patch("boto3.client")
    def test_get_ai_client_factory(self, mock_boto, mock_openai):
        bedrock = get_ai_client("bedrock")
        self.assertIsInstance(bedrock, BedrockClient)
        self.assertEqual(bedrock.model, "amazon.nova-pro-v1:0")

        openai = get_ai_client("openai", api_key="dummy")
        self.assertIsInstance(openai, OpenAIClient)
        self.assertEqual(openai.model, "gpt-5.4-nano")

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
        mock_response.model = "gpt-5.4-nano"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        ai_prompt.current_prompt = "Look around"
        actions = client.generate_actions()

        self.assertEqual(len(actions), 2)
        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "usage" for a in actions))
        narrative_actions = [a for a in actions if isinstance(a, NarrativeMessage)]
        self.assertEqual(len(narrative_actions), 1)
        self.assertEqual(narrative_actions[0].message, "You awaken in a dark room.")

        # Verify system message in chat completions call
        mock_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_instance.chat.completions.create.call_args.kwargs
        self.assertIn("messages", call_kwargs)
        self.assertEqual(call_kwargs["messages"][0], {"role": "system", "content": ai_prompt.SYSTEM_PROMPT})
        self.assertNotIn("system", call_kwargs)

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
        mock_response.model = "gpt-5.4-nano"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertEqual(len(actions), 4) # usage VerboseMessage, NarrativeMessage, toolUse VerboseMessage, EndGame
        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "usage" for a in actions))
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
        resp1 = MagicMock(model="gpt-5.4-nano", usage={"prompt_tokens": 10}, choices=[MagicMock(message=mock_bad_msg)])

        # 2nd response: fixed tool call
        mock_good_tool = MagicMock()
        mock_good_tool.id = "call_2"
        mock_good_tool.function.name = "end_game"
        mock_good_tool.function.arguments = '{"won": false}'
        mock_good_msg = MagicMock(content="Game lost", tool_calls=[mock_good_tool], reasoning_content=None, thinking=None)
        resp2 = MagicMock(model="gpt-5.4-nano", usage={"prompt_tokens": 10}, choices=[MagicMock(message=mock_good_msg)])

        mock_instance.chat.completions.create.side_effect = [resp1, resp2]

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "warn" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and not a.won for a in actions))

    @patch("openai.OpenAI")
    def test_openai_thinking_trace_and_history_exclusion(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        mock_msg = MagicMock()
        mock_msg.content = "You see a chest."
        mock_msg.reasoning_content = "Pondering options: chest might be trapped."
        mock_msg.tool_calls = None

        mock_response = MagicMock(model="gpt-5.4-nano", usage={"prompt_tokens": 10}, choices=[MagicMock(message=mock_msg)])
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        # Check that VerboseMessage was generated with reasoning text
        thinking_msgs = [a for a in actions if isinstance(a, VerboseMessage) and a.type == "thinking"]
        self.assertEqual(len(thinking_msgs), 1)
        self.assertEqual(thinking_msgs[0].message, "Pondering options: chest might be trapped.")

        # Check that mock_msg is recorded in client.messages
        self.assertIn(mock_msg, client.messages)


class TestBedrockClient(unittest.TestCase):

    @patch("boto3.client")
    def test_bedrock_generate_actions(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_boto.return_value = mock_bedrock

        mock_response = {
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
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
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "tu_1", "name": "end_game", "input": {"bad_arg": 1}}}
                    ]
                }
            }
        }

        resp2 = {
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
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
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
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
        self.assertEqual(thinking_msgs[0].message, repr(thinking_block))

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
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
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
