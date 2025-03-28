# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from argparse import ArgumentParser

from anemoi.utils.cli import cli_main
from anemoi.utils.cli import make_parser

from anemoi.transform import __version__
from anemoi.transform.commands import COMMANDS


# For read-the-docs
def create_parser() -> ArgumentParser:
    """Creates the argument parser for the CLI.

    Returns
    -------
    ArgumentParser
        The argument parser.
    """
    return make_parser(__doc__, COMMANDS)


def main() -> None:
    """The main entry point for the CLI."""
    cli_main(__version__, __doc__, COMMANDS)


if __name__ == "__main__":
    main()
