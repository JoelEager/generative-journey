import unittest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame
from generative_journey.cli import main


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
