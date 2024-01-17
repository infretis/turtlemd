import pytest
from pint import Quantity
from scipy.constants import Avogadro as AVOGADRO  # type: ignore
from scipy.constants import Boltzmann as BOLTZMANN

from turtlemd.units import UnitSystem


def test_units(capsys):
    """Test that we can create create a unit system with correct time."""
    unit = UnitSystem(
        name="gromacs",
        length=Quantity(1.0, "nm"),
        mass=Quantity(1.0 / AVOGADRO, "g"),
        energy=Quantity(1.0 / AVOGADRO, "kJ"),
        boltzmann=(Quantity(BOLTZMANN, "J/K").to("kJ/K") * AVOGADRO).magnitude,
        input_time_unit="ps",
    )
    assert pytest.approx(unit.time_factor) == 1.0
    assert pytest.approx(unit.boltzmann) == 0.00831446261815324
    print(unit)
    captured = capsys.readouterr()
    assert "gromacs" in captured.out
