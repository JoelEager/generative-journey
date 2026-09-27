"""Module defining constants and current state used to prompt the AI on each turn."""

SYSTEM_PROMPT = """You are the Game Master for an immersive text adventure game.
Your goal is to guide the player through an engaging story based on their choices.

Rules:
1. Provide vivid, evocative narrative descriptions to the player.
2. If the game reaches a definitive victory or defeat condition:
   - You MUST invoke the `end_game` tool with `won` set to true (if player won) or false (if player lost).
   - In a separate narrative message, describe why the game ended and what happened.
3. Keep turns concise and wait for player input unless the game has ended."""

TOOLS = [
    {
        "name": "end_game",
        "description": "Invoke when the game is won or lost.",
        "parameters": {
            "type": "object",
            "properties": {
                "won": {
                    "type": "boolean",
                    "description": "True if the player won the game, False if the player lost.",
                }
            },
            "required": ["won"],
        },
    }
]

current_prompt: str = ""

def intro_prompt(goal: str, transport: str) -> str:
    """Generate the initial prompt for the AI based on the player's goal and transport method."""
    return (
        f"The player has set out on a journey with the goal of '{goal}' "
        f"and intends to travel by '{transport}'. "
        f"Guide them through an engaging story based on their choices."
    )
