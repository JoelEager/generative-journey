import unittest

from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame, parse_tool_action


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


if __name__ == "__main__":
    unittest.main()
