"""A collection of simple position dependent potentials.

This module defines some potential functions which are useful
as simple models:

* DoubleWell (:py:class:`.DoubleWell`)
  This class defines a one-dimensional double well potential.

* RectangularWell (:py:class:`.RectangularWell`)
  This class defines a one-dimensional rectangular well potential.

* DoubleWellPair (:py:class:`.DoubleWellPair`)
    This class defines a double well pair potential.
"""
import logging

import numpy as np

from turtlemd.potentials.potential import Potential
from turtlemd.system.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class DoubleWell(Potential):
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
        params (dict): Contains the parameters. The keys are:

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
        """Initialise the one dimensional double well potential."""
        super().__init__(dim=1, desc=desc)
        self.params = {"a": a, "b": b, "c": c}

    def potential(self, system: System) -> float:
        """Evaluate the 1D double well potential."""
        pos = system.particles.pos
        v_pot = (
            self.params["a"] * pos**4
            - self.params["b"] * (pos - self.params["c"]) ** 2
        )
        return v_pot.sum()

    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the force for the 1D double well."""
        pos = system.particles.pos
        forces = -4.0 * (self.params["a"] * pos**3) + 2.0 * (
            self.params["b"] * (pos - self.params["c"])
        )
        # Set virial to zero - we do not compute it here.
        return forces, np.zeros_like(system.particles.virial)


class RectangularWell(Potential):
    r"""A 1D rectangular well potential.

    This class defines a one-dimensional rectangular well potential.
    The potential energy is zero within the potential well and infinite
    outside. The well is defined with a left and right boundary.

    Attributes
        params (dict): The parameters for the potential. The keys are:

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
        """Initialise the one-dimensional rectangular well."""
        super().__init__(dim=1, desc=desc)
        self.params = {
            "left": left,
            "right": right,
            "largenumber": largenumber,
        }
        if self.params["left"] >= self.params["right"]:
            msg = "Setting left >= right in RectangularWell potential!"
            LOGGER.warning(msg)

    def potential(self, system: System) -> float:
        """Evaluate the potential."""
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
        LOGGER.warning("Calling force for %s is not implemented!", self.desc)
        force = np.zeros_like(system.particles.force)
        virial = np.zeros_like(system.particles.virial)
        return force, virial


class DoubleWellPair(Potential):
    r"""A double well potential.

    This class defines a double well pair potential. The potential energy
    (:math:`V_\text{pot}`) for a pair of particles separated by a
    distance :math:`r` is given by,

    .. math::

       V_\text{pot} = h (1 - (r - r_0 - w)^2/w^2)^2,

    where :math:`h` gives the 'height' of the potential, :math:`r_0` the
    minimum and :math:`w` the width. These parameters are stored in the
    attributes `height`, `rzero` and `width` respectively.

    Attributes
        params (dict): Contains the parameters. These are:

            * `height`: A float describing the "height" of the potential.

            * `height4`: A float equal to ``4.0 * height``.
              (This variable is just included for convenience).

            * `rzero`: A float defining the two minimums. One is located at
              ``rzero``, the other at ``rzero+2*width``.


            * `width`: A float describing the "width" of the potential.

            * `width2`: A float equal to ``width*width`` (for convenience).

        types (tuple of ints): Indices defining what particles types
            the potential is active for.
    """

    types: tuple[int, int]

    def __init__(
        self,
        types: tuple[int, int],
        dim: int = 3,
        desc: str = "A double well pair potential",
    ):
        """Initialise the potential."""
        super().__init__(dim=dim, desc=desc)
        self.params = {
            "height": 0.0,
            "height4": 0.0,
            "rwidth": 0.0,
            "rzero": 0.0,
            "width": 0.0,
            "width2": 0.0,
        }
        self.types = types

    def set_parameters(self, parameters: dict[str, float]):
        """Add new potential parameters to the potential.

        Args:
            parameters (dict): The new parameters, they are assume to be
            dicts on the form:
            ``{'rzero': 1.0, 'width': 0.25, 'height': 6.0}``.

        """
        for key in parameters:
            if key in self.params:
                self.params[key] = parameters[key]
            else:
                msg = 'Ignored unknown parameter "%s"'
                LOGGER.warning(msg, key)
        self.params["width2"] = self.params["width"] ** 2
        self.params["rwidth"] = self.params["rzero"] + self.params["width"]
        self.params["height4"] = 4.0 * self.params["height"]

    def min_max(self) -> tuple[float, float, float]:
        """Return the minima & maximum of the potential.

        The minima are located at ``rzero`` & ``rzero + 2*width``.
        The maximum is located at ``rzero + width``.

        Returns:
            out[0]: Minimum number one, located at: ``rzero``.
            out[1]: Minimum number two, located at: ``rzero + 2*width``.
            out[2]: Maximum, located at: ``rzero + width``.

        """
        rzero = self.params["rzero"]
        width = self.params["width"]
        return rzero, rzero + 2.0 * width, rzero + width

    def activate(self, itype: int, jtype: int) -> bool:
        """Determine if we should calculate a interaction or not."""
        return (itype == self.types[0] and jtype == self.types[1]) or (
            jtype == self.types[0] and itype == self.types[1]
        )

    def potential(self, system: System) -> float:
        """Calculate the potential energy."""
        particles = system.particles
        box = system.box
        v_pot = 0.0

        for pair in particles.pairs():
            i, j, itype, jtype = pair
            if self.activate(itype, jtype):
                delta = box.pbc_dist(particles.pos[i] - particles.pos[j])
                delr = np.sqrt(np.dot(delta, delta))
                v_pot += self._potential_function(delr)
        return v_pot

    def _potential_function(self, delr: float) -> float:
        """Calculate the potential.

        This method can be used to visualize the potential energy as a
        function of the bond length.

        """
        rwidth = self.params["rwidth"]
        width2 = self.params["width2"]
        height = self.params["height"]
        return height * (1.0 - (((delr - rwidth) ** 2) / width2)) ** 2

    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the force on the particles."""
        particles = system.particles
        box = system.box
        forces = np.zeros(particles.pos.shape)
        virial = np.zeros((particles.dim, particles.dim))

        rwidth = self.params["rwidth"]
        width2 = self.params["width2"]
        height4 = self.params["height4"]

        for pair in particles.pairs():
            i, j, itype, jtype = pair
            if self.activate(itype, jtype):
                delta = box.pbc_dist(particles.pos[i] - particles.pos[j])
                delr = np.sqrt(np.dot(delta, delta))
                diff = delr - rwidth
                forceij = (
                    height4 * (1.0 - diff**2 / width2) * (diff / width2)
                )
                forceij = forceij * delta / delr
                forces[i] += forceij
                forces[j] -= forceij
                virial += np.outer(forceij, delta)
        return forces, virial
