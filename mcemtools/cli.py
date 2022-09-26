# -*- coding: utf-8 -*-

"""Console script for mcemtools."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for mcemtools."""
    click.echo("Welcome to mcemtools")
    click.echo("A Python package to run state-of-the-art solutions for" + \
               " electron microscopy problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover