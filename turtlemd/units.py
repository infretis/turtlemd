"""Define unit conversions for TurtleMD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pint import UnitRegistry
from scipy.constants import Avogadro as AVOGADRO  # type: ignore
from scipy.constants import Boltzmann as BOLTZMANN

if TYPE_CHECKING:  # pragma: no cover
    from pint.facets.plain.quantity import PlainQuantity

Quantity = UnitRegistry().Quantity


def time_unit(
    length: PlainQuantity, mass: PlainQuantity, energy: PlainQuantity
) -> PlainQuantity:
    """Calculate the time unit given the length, mass, and energy units.

    A system of units will typically define the length (L), mass (M),
    and time (T) units. The energy unit (E) is then given as E = ML²/T².
    With MD, it is often convenient to define the length, mass, and
    energy units. The time unit is then given by: T = (ML²/E)^(1/2).
    This method calculates the time unit as stated above and returns
    the answer in fs.

    Args:
        length: The length unit.
        mass: The mass unit.
        energy: The energy unit.

    Returns:
        The time unit in fs.
    """
    time_s = np.sqrt(length.to("m") ** 2 * mass.to("kg") / energy.to("J"))
    return Quantity(time_s.magnitude, "s").to("fs")


class UnitSystem:
    """Define a simple unit system."""

    def __init__(
        self,
        name: str,
        length: PlainQuantity,
        mass: PlainQuantity,
        energy: PlainQuantity,
        boltzmann: float,
        input_time_unit: str = "internal",
    ):
        """Set up the unit system.

        In particular, the time unit is calculated here.
        """
        self.name = name
        self.base_units = {
            "length": length,
            "mass": mass,
            "energy": energy,
        }
        self.time_fs = time_unit(length, mass, energy)
        self.boltzmann = boltzmann
        if input_time_unit == "internal":
            self.time_factor = 1.0
        else:
            self.time_factor = 1.0 / self.time_fs.to(input_time_unit).magnitude

    def __str__(self):
        """Return information on values and units as text."""
        msg = [
            f'# Unit system "{self.name}"',
            f"\t* Time: {self.time_fs}",
            f"\t* Time factor: {self.time_factor}",
            f"\t* Boltzmann: {self.boltzmann}",
        ]
        for key, val in self.base_units.items():
            msg.append(f"\t* {key.capitalize()}: {val}")
        return "\n".join(msg)


UNIT_SYSTEMS: dict[str, UnitSystem] = {}

UNIT_SYSTEMS["reduced"] = UnitSystem(
    name="reduced",
    length=Quantity(1.0, "Å"),
    mass=Quantity(1.0 / AVOGADRO, "g"),
    energy=Quantity(BOLTZMANN, "J"),
    boltzmann=1.0,
    input_time_unit="internal",
)

UNIT_SYSTEMS["lj"] = UnitSystem(
    name="lj",
    length=Quantity(3.405, "Å"),
    mass=Quantity(39.948 / AVOGADRO, "g"),
    energy=Quantity(BOLTZMANN, "J") * 119.8,
    boltzmann=1.0,
    input_time_unit="internal",
)

UNIT_SYSTEMS["real"] = UnitSystem(
    name="real",
    length=Quantity(1.0, "Å"),
    mass=Quantity(1.0 / AVOGADRO, "g"),
    energy=Quantity(1.0 / AVOGADRO, "kcal"),
    boltzmann=(Quantity(BOLTZMANN, "J/K").to("kcal/K") * AVOGADRO).magnitude,
    input_time_unit="fs",
)
