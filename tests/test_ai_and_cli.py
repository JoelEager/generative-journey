import unittest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame, parse_tool_action
from generative_journey.ai_clients import BaseAIClient, get_ai_client
from generative_journey.ai_clients.anthropic import AnthropicClient
from generative_journey.ai_clients.bedrock import BedrockClient
from generative_journey.ai_clients.local import LocalClient
from generative_journey.ai_clients.openai import OpenAIClient
from generative_journey.cli import main


class TestAIActionsAndPrompt(unittest.TestCase):

    def test_narrative_message(self):
        msg = NarrativeMessage("Hello adventure")
        self.assertEqual(msg.message, "Hello adventure")
        self.assertEqual(str(msg), "Hello adventure")

    def test_verbose_message(self):
        msg1 = VerboseMessage("Analyzing decision")
        self.assertEqual(str(msg1), "thinking: Analyzing decision")

        msg2 = VerboseMessage("Invalid tool call", type="warn")
        self.assertEqual(str(msg2), "warn: Invalid tool call")

    def test_end_game(self):
        end_win = EndGame(won=True)
        self.assertTrue(end_win.won)
        end_loss = EndGame(won=False)
        self.assertFalse(end_loss.won)

    def test_parse_tool_action_valid(self):
        action = parse_tool_action("end_game", {"won": True})
        self.assertIsInstance(action, EndGame)
        self.assertTrue(action.won)

        action_loss = parse_tool_action("end_game", {"won": False})
        self.assertIsInstance(action_loss, EndGame)
        self.assertFalse(action_loss.won)

    def test_parse_tool_action_invalid_args(self):
        with self.assertRaises(ValueError):
            parse_tool_action("end_game", "not_a_dict")

        with self.assertRaises(ValueError):
            parse_tool_action("end_game", {})

        with self.assertRaises(ValueError):
            parse_tool_action("end_game", {"won": "yes"})

    def test_parse_tool_action_unknown_tool(self):
        with self.assertRaises(ValueError):
            parse_tool_action("unknown_tool", {"won": True})


class DummyAIClient(BaseAIClient):
    provider_name = "dummy"

    def _format_tools(self, tools):
        return tools

    def _call_model(self, formatted_tools):
        return None

    def _parse_response(self, response):
        return [], [], False, [], []


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


class TestAIClients(unittest.TestCase):

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

    @patch("openai.OpenAI")
    def test_openai_generate_actions_success(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        mock_msg = MagicMock()
        mock_msg.content = "You awaken in a dark room."
        mock_msg.tool_calls = None
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
        mock_tool_call.function.name = "end_game"
        mock_tool_call.function.arguments = '{"won": true}'

        mock_msg = MagicMock()
        mock_msg.content = "Victory is yours!"
        mock_msg.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertEqual(len(actions), 2)
        self.assertIsInstance(actions[0], NarrativeMessage)
        self.assertIsInstance(actions[1], EndGame)
        self.assertTrue(actions[1].won)

    @patch("openai.OpenAI")
    def test_openai_generate_actions_invalid_tool_retry(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        # 1st response: invalid arguments
        mock_bad_tool = MagicMock()
        mock_bad_tool.id = "call_1"
        mock_bad_tool.function.name = "end_game"
        mock_bad_tool.function.arguments = '{"invalid_param": 123}'
        mock_bad_msg = MagicMock(content="Attempting end game", tool_calls=[mock_bad_tool])
        resp1 = MagicMock(choices=[MagicMock(message=mock_bad_msg)])

        # 2nd response: fixed tool call
        mock_good_tool = MagicMock()
        mock_good_tool.id = "call_2"
        mock_good_tool.function.name = "end_game"
        mock_good_tool.function.arguments = '{"won": false}'
        mock_good_msg = MagicMock(content="Game lost", tool_calls=[mock_good_tool])
        resp2 = MagicMock(choices=[MagicMock(message=mock_good_msg)])

        mock_instance.chat.completions.create.side_effect = [resp1, resp2]

        client = OpenAIClient(api_key="dummy")
        actions = client.generate_actions()

        self.assertTrue(any(isinstance(a, VerboseMessage) and a.type == "warn" for a in actions))
        self.assertTrue(any(isinstance(a, EndGame) and not a.won for a in actions))

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

        self.assertEqual(len(actions), 2)
        self.assertIsInstance(actions[0], NarrativeMessage)
        self.assertIsInstance(actions[1], EndGame)
        self.assertTrue(actions[1].won)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy_anthropic_key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_generate_actions_invalid_tool_retry(self, mock_anthropic_cls):
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        # 1st response: invalid arguments
        mock_bad_tool = MagicMock()
        mock_bad_tool.type = "tool_use"
        mock_bad_tool.id = "tool_1"
        mock_bad_tool.name = "end_game"
        mock_bad_tool.input = {"invalid_param": 123}
        resp1 = MagicMock(content=[mock_bad_tool])

        # 2nd response: valid tool call
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

        self.assertEqual(len(actions), 2)
        self.assertIsInstance(actions[0], NarrativeMessage)
        self.assertIsInstance(actions[1], EndGame)
        self.assertTrue(actions[1].won)

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


class TestCLI(unittest.TestCase):

    @patch("generative_journey.cli.get_ai_client")
    def test_cli_game_loop(self, mock_get_ai_client):
        mock_client = MagicMock()
        mock_client.__repr__ = MagicMock(return_value="provider: openai, model: gpt-4o")

        # Turn 1: Narrative turn
        # Turn 2: EndGame turn
        turn1_actions = [NarrativeMessage("Welcome to the labyrinth.")]
        turn2_actions = [NarrativeMessage("A dragon devours you."), EndGame(won=False)]
        mock_client.generate_actions.side_effect = [turn1_actions, turn2_actions]

        mock_get_ai_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["openai"], input="\n\ngo north\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Welcome to the labyrinth.", result.output)
        self.assertIn("A dragon devours you.", result.output)
        self.assertIn("You lost the game!", result.output)

    @patch("generative_journey.cli.get_ai_client")
    def test_cli_verbose_option(self, mock_get_ai_client):
        mock_client = MagicMock()
        mock_client.__repr__ = MagicMock(return_value="provider: local")

        turn_actions = [
            VerboseMessage("Thinking deeply...", type="thinking"),
            NarrativeMessage("You see a shiny key."),
            EndGame(won=True)
        ]
        mock_client.generate_actions.return_value = turn_actions
        mock_get_ai_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["local", "-v"], input="\n\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("thinking: Thinking deeply...", result.output)
        self.assertIn("You see a shiny key.", result.output)
        self.assertIn("You won the game!", result.output)


if __name__ == "__main__":
    unittest.main()
