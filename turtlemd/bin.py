"""Command line execution of TurtleMD."""
import argparse
import logging

from turtlemd.inout.settings import (
    create_integrator_from_settings,
    create_system_from_settings,
    read_settings_file,
)
from turtlemd.simulation import MDSimulation
from turtlemd.version import __version__

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


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

    LOGGER.info(f"Reading settings from file {args.input_file}")
    settings = read_settings_file(args.input_file)
    system = create_system_from_settings(settings)
    LOGGER.info(f"Created system: {system}")
    integrator = create_integrator_from_settings(settings)
    LOGGER.info(f"Created integrator {integrator}")

    simulation = MDSimulation(
        system=system,
        integrator=integrator,
        steps=settings["md"]["steps"],
        start=0,
    )

    return settings, simulation
