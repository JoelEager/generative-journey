"""CLI entry point for Generative Journey."""
from traceback import format_exc
import click

from . import ai_prompt
from .ai_actions import NarrativeMessage, VerboseMessage, EndGame
from .ai_clients import get_ai_client


def fatal_error(message: str):
    """Prints a stack trace and exits the program with an error message."""
    click.secho("\n" + format_exc(), err=True, fg="red")
    raise click.ClickException(message)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("provider")
@click.option("-m", "--model", help="AI model name override.")
@click.option("-v", "--verbose", is_flag=True, help="Display thinking and other messages.")
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

    goal = click.prompt("What is your goal for this adventure?", default="Find the hidden treasure")
    transport = click.prompt("How do you intend to get there?", default="On foot")
    ai_prompt.current_prompt = ai_prompt.intro_prompt(goal, transport)

    game_over = False
    while not game_over:
        try:
            actions = client.generate_actions()
        except Exception:
            fatal_error("Failure during AI invocation")

        for action in actions:
            if isinstance(action, VerboseMessage):
                if verbose:
                    click.secho(str(action), fg="yellow")
            elif isinstance(action, NarrativeMessage):
                click.secho("\n" + str(action), fg="cyan")
            elif isinstance(action, EndGame):
                game_over = True
                if action.won:
                    click.secho("\nYou won the game!", fg="bright_green", italic=True)
                else:
                    click.secho("\nYou lost the game!", fg="bright_red", italic=True)

        if game_over:
            break

        ai_prompt.current_prompt = click.prompt("Your action")


if __name__ == "__main__":
    main()
