"""Actions returned by AI clients during game turns."""
from typing import Any


class NarrativeMessage:
    """Represents a narrative message to be presented to the player."""

    def __init__(self, message: str):
        self.message = message.strip()

    def __str__(self) -> str:
        return self.message


class VerboseMessage:
    """Represents a message intended for verbose mode."""

    def __init__(self, message: str, type: str = "thinking"):
        self.message = message.strip()
        self.type = type

    def __str__(self) -> str:
        return f"{self.type}: {self.message}"


class EndGame:
    """Represents an invocation of the end_game tool when the game is won or lost."""

    def __init__(self, won: bool):
        self.won = bool(won)


def parse_tool_action(fn_name: str, args: Any) -> Any:
    """Validate tool call arguments and return corresponding action object.

    Raises ValueError if args is not a dict, missing required parameters, has invalid parameter types,
    or has an unknown function name.
    """
    if not isinstance(args, dict):
        raise ValueError(f"Invalid arguments for {fn_name}: {args}")

    if fn_name == "end_game":
        if "won" not in args or not isinstance(args["won"], bool):
            raise ValueError(f"Invalid arguments for {fn_name}: {args}")
        return EndGame(won=args["won"])

    raise ValueError(f"Unknown tool name: {fn_name}")
