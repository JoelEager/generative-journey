import sys
import click

from generative_journey.ai_clients import SUPPORTED_PROVIDERS, get_ai_client


@click.command()
@click.argument("provider", required=False)
@click.option("-m", "--model", help="AI model name override.")
def main(provider: str | None, model: str | None):
    """Main entry point for Generative Journey CLI."""
    if not provider:
        click.echo("Error: Missing required argument 'PROVIDER'.", err=True)
        click.echo(
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}.\n"
            "Please refer to README.md for details on supported providers, models, and API key configurations.",
            err=True,
        )
        sys.exit(1)

    provider_clean = provider.lower().strip()
    if provider_clean not in SUPPORTED_PROVIDERS:
        click.echo(f"Error: Unsupported AI provider '{provider}'.", err=True)
        click.echo(
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}.\n"
            "Please refer to README.md for details on supported providers, models, and API key configurations.",
            err=True,
        )
        sys.exit(1)

    try:
        client = get_ai_client(provider_clean, model=model)
        click.echo(f"Welcome to Generative Journey! ({client})")
        prompt = "Introduce a mystical adventure game setting in two evocative sentences."
        response = client.generate_response(prompt)
        click.echo(f"\nAI Response:\n{response}")
    except Exception as e:
        click.echo(f"Error communicating with AI service: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
