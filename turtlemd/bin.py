"""Command line execution of TurtleMD."""
import argparse

from turtlemd.inout.settings import (
    create_system_from_settings,
    read_settings_file,
)
from turtlemd.version import __version__


def main():
    """Set up the parser and read the input file."""
    parser = argparse.ArgumentParser(
        # prog="TurtleMD",
        description="Run TurtleMD from input files.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        help="path to the input TOML file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program version and exit.",
    )
    args = parser.parse_args()

    settings = read_settings_file(args.input_file)
    system = create_system_from_settings(settings)
    print(system)

    return settings
