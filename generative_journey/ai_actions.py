"""Actions returned by AI clients during game turns."""

class NarrativeMessage:
    """Represents a narrative message to be presented to the player."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"NarrativeMessage({self.message!r})"


class VerboseMessage:
    """Represents a message intended for verbose mode (e.g., thinking or warn)."""

    def __init__(self, message: str, type: str = "thinking"):
        self.message = message
        self.type = type

    def __str__(self) -> str:
        return f"{self.type}: {self.message}"

    def __repr__(self) -> str:
        return f"VerboseMessage({self.message!r}, type={self.type!r})"


class EndGame:
    """Represents an invocation of the end_game tool when the game is won or lost."""

    def __init__(self, won: bool):
        self.won = bool(won)

    def __repr__(self) -> str:
        return f"EndGame(won={self.won})"
