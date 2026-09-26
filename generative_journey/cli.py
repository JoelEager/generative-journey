"""CLI entry point for Generative Journey."""
from traceback import format_exc
import click

from generative_journey import ai_prompt
from generative_journey.ai_actions import NarrativeMessage, VerboseMessage, EndGame
from generative_journey.ai_clients import get_ai_client


def fatal_error(message: str):
    """Prints a stack trace and exits the program with an error message."""
    click.secho("\n" + format_exc(), err=True, fg="red")
    raise click.ClickException(message)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("provider")
@click.option("-m", "--model", help="AI model name override.")
@click.option("-v", "--verbose", is_flag=True, help="Display verbose AI messages (thinking, warnings, etc.).")
def main(provider, model, verbose):
    """
    Play a game of Generative Journey with the specified AI. See the readme for information on supported providers and models.
    """
    provider = provider.lower().strip()
    kwargs = {}
    if model is not None:
        kwargs["model"] = model

    try:
        client = get_ai_client(provider, **kwargs)
    except ValueError as e:
        raise click.UsageError(str(e))
    except Exception:
        fatal_error("Failed to initialize AI client")

    click.echo(f"Welcome to Generative Journey! ({client})\n")
    ai_prompt.PLAYER_MESSAGE = ""

    game_over = False

    while not game_over:
        try:
            actions = client.generate_actions()
        except Exception:
            fatal_error("Failure during AI interaction")

        for action in actions:
            if isinstance(action, VerboseMessage):
                if verbose:
                    click.echo(str(action))
            elif isinstance(action, NarrativeMessage):
                click.echo(str(action))
            elif isinstance(action, EndGame):
                game_over = True
                if action.won:
                    click.secho("\n*** VICTORY! You won the game! ***", fg="green", bold=True)
                else:
                    click.secho("\n*** GAME OVER! You lost the game! ***", fg="red", bold=True)

        if game_over:
            break

        ai_prompt.PLAYER_MESSAGE = click.prompt("\nYour action")


if __name__ == "__main__":
    main()
