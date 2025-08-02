"""Main function for Graphomotor."""

from graphomotor.core import cli


def run_main() -> None:
    """Main entry point to Graphomotor."""
    cli.app()


if __name__ == "__main__":
    cli.app()
