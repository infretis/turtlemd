"""A collection of simple position dependent potentials.

This module defines some potential functions which are useful
as simple models:

* DoubleWell (:py:class:`.DoubleWell`)
  This class defines a one-dimensional double well potential.

* RectangularWell (:py:class:`.RectangularWell`)
  This class defines a one-dimensional rectangular well potential.
"""
import logging

import numpy as np

from turtlemd.potentials.potential import PotentialFunction
from turtlemd.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class DoubleWell(PotentialFunction):
    r"""A 1D double well potential.

    This class defines a one-dimensional double well potential.
    The potential energy (:math:`V_\text{pot}`) is given by

    .. math::

       V_\text{pot} = a x^4 - b (x - c)^2

    where :math:`x` is the position and :math:`a`, :math:`b`
    and :math:`c` are parameters for the potential. These parameters
    are stored as attributes of the class. Typically, both :math:`a`
    and :math:`b` are positive quantities, however, we do not explicitly
    check that here.

    Attributes
    ----------
    params : dict
        Contains the parameters. The keys are:

        * `a`: The ``a`` parameter for the potential.
        * `b`: The ``b`` parameter for the potential.
        * `c`: The ``c`` parameter for the potential.

        These keys corresponds to the parameters in the potential,
        :math:`V_\text{pot} = a x^4 - b (x - c)^2`.

    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.0,
        desc: str = "1D double well potential",
    ):
        """Initialise the one dimensional double well potential.

        Parameters
        ----------
        a : float, optional
            Parameter for the potential.
        b : float, optional
            Parameter for the potential.
        c : float, optional
            Parameter for the potential.
        desc : string, optional
            Description of the force field.

        """
        super().__init__(dim=1, desc=desc)
        self.params = {"a": a, "b": b, "c": c}

    def potential(self, system: System) -> float:
        """Evaluate the potential for the one-dimensional double well.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we evaluate the potential for.

        Returns
        -------
        out : float
            The potential energy.

        """
        pos = system.particles.pos
        v_pot = (
            self.params["a"] * pos**4
            - self.params["b"] * (pos - self.params["c"]) ** 2
        )
        return v_pot.sum()

    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate forces for the 1D double well potential.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we evaluate the potential for.

        Returns
        -------
        out[0] : numpy.array
            The calculated force.
        out[1] : numpy.array
            The virial - not implemented for this potential!

        """
        pos = system.particles.pos
        forces = -4.0 * (self.params["a"] * pos**3) + 2.0 * (
            self.params["b"] * (pos - self.params["c"])
        )
        virial = np.zeros((self.dim, self.dim))  # just return zeros here
        return forces, virial

    def potential_and_force(
        self, system: System
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate the potential and the force.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we evaluate the potential for.

        Returns
        -------
        out[0] : float
            The potential energy as a float.
        out[1] : numpy.array
            The force as a numpy.array of the same shape as the
            positions in `particles.pos`.
        out[2] : numpy.array
            The virial - not implemented for this potential!

        """
        pos = system.particles.pos
        dist = pos - self.params["c"]
        pos3 = pos**3
        v_pot = self.params["a"] * pos3 * pos - self.params["b"] * dist**2
        forces = -4.0 * (self.params["a"] * pos3) + 2.0 * (
            self.params["b"] * dist
        )
        virial = np.zeros((self.dim, self.dim))  # just return zeros here
        return v_pot.sum(), forces, virial


class RectangularWell(PotentialFunction):
    r"""A 1D rectangular well potential.

    This class defines a one-dimensional rectangular well potential.
    The potential energy is zero within the potential well and infinite
    outside. The well is defined with a left and right boundary.

    Attributes
    ----------
    params : dict
        The parameters for the potential. The keys are:

        * `left`: Left boundary of the potential.
        * `right`: Right boundary of the potential.
        * `largenumber`: Value of potential outside the boundaries.

        It is possible to define left > right, however, a warning will
        be issued then.

    """

    def __init__(
        self,
        left: float = 0.0,
        right: float = 1.0,
        largenumber: float = float("inf"),
        desc: str = "1D Rectangular well potential",
    ):
        """Initialise the one-dimensional rectangular well.

        Parameters
        ----------
        left : float, optional
            The left boundary of the potential.
        right : float, optional
            The right boundary of the potential.
        largenumber : float, optional
            The value of the potential outside (left, right).
        desc : string, optional
            Description of the force field.

        """
        super().__init__(dim=1, desc=desc)
        self.params = {
            "left": left,
            "right": right,
            "largenumber": largenumber,
        }
        self.check_parameters()

    def check_parameters(self) -> bool:
        """Check the consistency of the parameters."""
        if self.params["left"] >= self.params["right"]:
            msg = "Setting left >= right in RectangularWell potential!"
            LOGGER.warning(msg)
        return True

    def potential(self, system: System) -> float:
        """Evaluate the potential.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out : float
            The potential energy.

        """
        pos = system.particles.pos
        left = self.params["left"]
        right = self.params["right"]
        largenumber = self.params["largenumber"]
        v_pot = np.where(
            np.logical_and(pos > left, pos < right), 0.0, largenumber
        )
        return v_pot.sum()

    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Not implemented, just return zeros."""
        LOGGER.warning("Calling force for {self.desc} is not implemented!")
        force = np.zeros_like(system.particles.force)
        virial = np.zeros_like(system.particles.virial)
        return force, virial
