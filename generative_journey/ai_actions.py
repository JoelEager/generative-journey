"""Actions returned by AI clients during game turns."""

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
