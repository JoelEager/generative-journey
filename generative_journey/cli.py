"""CLI entry point for Generative Journey."""
from traceback import format_exc
import click

from .ai_clients import get_ai_client


def fatal_error(message: str):
    """Prints a stack trace and exits the program with an error message."""
    click.secho("\n" + format_exc(), err=True, fg="red")
    raise click.ClickException(message)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("provider")
@click.option("-m", "--model", help="AI model name override.")
def main(provider, model):
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

    try:
        click.echo(f"Welcome to Generative Journey! ({client})")
        prompt = "Introduce a mystical adventure game setting in two evocative sentences."
        response = client.generate_response(prompt)
        click.echo(f"\nAI Response:\n{response}")
    except Exception:
        fatal_error("Failure during AI interaction")


if __name__ == "__main__":
    main()
