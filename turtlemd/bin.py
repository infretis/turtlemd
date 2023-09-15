"""Command line execution of TurtleMD."""
import argparse

from turtlemd.version import __version__


def main():
    parser = argparse.ArgumentParser(
        # prog="TurtleMD",
        description="Run TurtleMD from input files",
    )
    parser.add_argument(
        "-i", "--input_file", help="path to the input TOML file.", type=str
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program version and exit.",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
