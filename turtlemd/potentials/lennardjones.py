"""Module defining a Lennard-Jones pair potential."""
import itertools
import logging
from typing import Any

import numpy as np

from turtlemd.potentials.potential import Potential

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class LennardJonesCut(Potential):
    r"""Lennard-Jones 6-12 potential.

    The Lennard-Jones 6-12 potential (:math:`V_\text{pot}`)
    for an interacting pair of particles a distance :math:`r` apart,
    is here,

    .. math::

       V_\text{pot} = 4 \varepsilon \left( x^{12} - x^{6} \right),

    where :math:`x = \sigma/r`, and :math:`\varepsilon`
    and :math:`\sigma` are the potential parameters.
    We need one set of parameters for each pair of particle types.
    Parameters can be generated with a specific mixing rule.

    Attributes:
        params (dict): The parameters for the potential. This dict is assumed
            to contain parameters for pairs, i.e., for interactions.
            Parameters can be generated with a specific mixing rule.
        _lj1 (dict): Lennard-Jones parameters used for calculation of
            the force. Keys are the pairs (particle types) that may interact.
            Calculated as: 48.0 * epsilon * sigma**12.
        _lj2 (dict): Lennard-Jones parameters used for calculation of
            the force. Keys are the pairs (particle types) that may interact.
            Calculated as: 24.0 * epsilon * sigma**6.
        _lj3 (dict): Lennard-Jones parameters used for calculation of the
            potential. Keys are the pairs (particle types) that may interact.
            Calculated as: 4.0 * epsilon * sigma**12.
        _lj4 (dict): Lennard-Jones parameters used for calculation of the
            potential. Keys are the pairs (particle types) that may interact.
            Calculated as: 4.0 * epsilon * sigma**6.
        _offset (dict): Potential values for shifting the potential.
            This is the potential evaluated at the cut-off.
        _rcut2 (dict): The squared cut-off for each interaction type.
            Keys are the pairs (particle types) that may interact.
    """

    shift: bool
    _lj1: dict[tuple[int, int], float]
    _lj2: dict[tuple[int, int], float]
    _lj3: dict[tuple[int, int], float]
    _lj4: dict[tuple[int, int], float]
    _offset: dict[tuple[int, int], float]
    _rcut2: dict[tuple[int, int], float]

    def __init__(
        self,
        dim: int = 3,
        shift: bool = True,
        mixing: str = "geometric",
        desc: str = "Lennard-Jones pair potential",
    ):
        """Initialise the Lennard-Jones potential.

        Args:
            dim: The dimensionality to use.
            shift: Determines if the potential should be shifted or not.
            mixing: Determines how we should mix potential parameters.
            desc: Short description of the potential.

        """
        super().__init__(dim=dim, desc=desc)
        self.shift = shift
        self._lj1 = {}
        self._lj2 = {}
        self._lj3 = {}
        self._lj4 = {}
        self._rcut2 = {}
        self._offset = {}
        self.params = {}
        self.mixing = mixing

    def set_parameters(self, parameters: dict[Any, dict[str, float]]):
        """Set and update parameters.

        Args:
            parameters : The potential parameters.

        """
        self.params = {}
        pair_param = generate_pair_interactions(parameters, self.mixing)
        for pair in pair_param:
            eps_ij = pair_param[pair]["epsilon"]
            sig_ij = pair_param[pair]["sigma"]
            rcut = pair_param[pair]["rcut"]
            self._lj1[pair] = 48.0 * eps_ij * sig_ij**12
            self._lj2[pair] = 24.0 * eps_ij * sig_ij**6
            self._lj3[pair] = 4.0 * eps_ij * sig_ij**12
            self._lj4[pair] = 4.0 * eps_ij * sig_ij**6
            self._rcut2[pair] = rcut**2
            vcut = 0.0
            if self.shift:
                try:
                    vcut = (
                        4.0
                        * eps_ij
                        * ((sig_ij / rcut) ** 12 - (sig_ij / rcut) ** 6)
                    )
                except ZeroDivisionError:
                    vcut = 0.0
            self._offset[pair] = vcut
            self.params[pair] = pair_param[pair]

    def __str__(self):
        """Generate a string with the potential parameters.

        It will generate a string with both pair and atom parameters.

        Returns:
            out: Table with the parameters of all interactions.

        """
        strparam = [self.desc]
        strparam += ["Potential parameters, Lennard-Jones:"]
        useshift = "yes" if self.shift else "no"
        strparam.append(f"Shift potential: {useshift}")
        atmformat = "{0:12s} {1:>9s} {2:>9s} {3:>9s}"
        atmformat2 = "{0:12s} {1:>9.4f} {2:>9.4f} {3:>9.4f}"
        strparam.append("Pair parameters:")
        strparam.append(
            atmformat.format("Atom/pair", "epsilon", "sigma", "cut-off")
        )
        for pair in sorted(self.params):
            eps_ij = self.params[pair]["epsilon"]
            sig_ij = self.params[pair]["sigma"]
            rcut = np.sqrt(self._rcut2[pair])
            stri = f"{pair[0]}-{pair[1]}"
            strparam.append(atmformat2.format(stri, eps_ij, sig_ij, rcut))
        return "\n".join(strparam)

    def potential(self, system):
        """Calculate the potential energy for the Lennard-Jones interaction.

        Args:
            system: The system for which we calculate the potential.

        Returns:
            out: The potential energy as a float.

        """
        particles = system.particles
        box = system.box
        pot = 0.0
        # the particle list may implement a list which we can
        # loop over. This could be some kind of fancy neighbour list
        # here, we ignore this and loop over all pairs using numpy.
        for i, particle_i in enumerate(particles.pos[:-1]):
            itype = particles.ptype[i]
            delta = particle_i - particles.pos[i + 1 :]
            delta = box.pbc_dist_matrix(delta)
            rsq = np.einsum("ij, ij->i", delta, delta)
            k = np.where(
                _check_cutoff(
                    self._rcut2, rsq, particles.ptype[i + 1 :], itype
                )
            )[0]
            if len(k) > 0:
                r6inv = 1.0 / rsq[k] ** 3
                pot += np.sum(
                    _pot_term(
                        self._lj3,
                        self._lj4,
                        self._offset,
                        r6inv,
                        particles.ptype[k + i + 1],
                        itype,
                    )
                )
        return pot

    def force(self, system):
        """Calculate the Lennard-Jones force and virial.

        Args:
            system: The system for which we calculate the force.

        Returns:
            out[0]: The forces for the particles.
            out[1]: The virial for the system.

        """
        particles = system.particles
        forces = np.zeros(particles.pos.shape)
        virial = np.zeros((system.box.dim, system.box.dim))
        for i, particle_i in enumerate(particles.pos[:-1]):
            itype = particles.ptype[i]
            delta = particle_i - particles.pos[i + 1 :]
            delta = system.box.pbc_dist_matrix(delta)
            rsq = np.einsum("ij, ij->i", delta, delta)
            k = np.where(
                _check_cutoff(
                    self._rcut2, rsq, particles.ptype[i + 1 :], itype
                )
            )[0]
            if len(k) > 0:
                r2inv = 1.0 / rsq[k]
                r6inv = r2inv**3
                forcelj = _force_term(
                    self._lj1,
                    self._lj2,
                    r2inv,
                    r6inv,
                    particles.ptype[k + i + 1],
                    itype,
                )
                forceij = np.einsum("i,ij->ij", forcelj, delta[k])
                forces[i] += np.sum(forceij, axis=0)
                forces[k + i + 1] -= forceij
                virial += np.einsum("ij,ik->jk", forceij, delta[k])
        return forces, virial

    def potential_and_force(self, system):
        """Calculate the Lennard Jones potential, force and virial.

        Args:
            system: The system to evaluate potential/force for.

        Returns:
            out[0]: The potential energy as a float.
            out[1]: The force on the particles.
            out[2]: The virial for the system.
        """
        particles = system.particles
        box = system.box
        pot = 0.0
        forces = np.zeros(particles.pos.shape)
        virial = np.zeros((box.dim, box.dim))
        for i, particle_i in enumerate(particles.pos[:-1]):
            itype = particles.ptype[i]
            delta = particle_i - particles.pos[i + 1 :]
            delta = box.pbc_dist_matrix(delta)
            rsq = np.einsum("ij, ij->i", delta, delta)
            k = np.where(
                _check_cutoff(
                    self._rcut2, rsq, particles.ptype[i + 1 :], itype
                )
            )[0]
            if len(k) > 0:
                jtype = particles.ptype[k + i + 1]
                r2inv = 1.0 / rsq[k]
                r6inv = r2inv**3
                pot += np.sum(
                    _pot_term(
                        self._lj3,
                        self._lj4,
                        self._offset,
                        r6inv,
                        jtype,
                        itype,
                    )
                )
                forcelj = _force_term(
                    self._lj1, self._lj2, r2inv, r6inv, jtype, itype
                )
                forceij = np.einsum("i,ij->ij", forcelj, delta[k])
                forces[i] += np.sum(forceij, axis=0)
                forces[k + i + 1] -= forceij
                virial += np.einsum("ij,ik->jk", forceij, delta[k])
        return pot, forces, virial


@np.vectorize
def _pot_term(lj3, lj4, offset, r6inv, jtype, itype):
    """Lennard Jones potential term."""
    return (
        r6inv * (lj3[itype, jtype] * r6inv - lj4[itype, jtype])
        - offset[itype, jtype]
    )


@np.vectorize
def _force_term(lj1, lj2, r2inv, r6inv, jtype, itype):
    """Lennard Jones force term."""
    return r2inv * r6inv * (lj1[itype, jtype] * r6inv - lj2[itype, jtype])


@np.vectorize
def _check_cutoff(rcut2, rsq, jtype, itype):
    """Check if we are close than the cut-off."""
    return rsq < rcut2[itype, jtype]


def generate_pair_interactions(
    parameters: dict[Any, dict[str, float]], mixing: str
) -> dict[tuple[int, int], dict[str, float]]:
    """Generate pair parameters from atom parameters.

    The parameters are given as a dictionary where the keys are
    either just integers -- which defines atom parameters -- or tuples
    which define pair interactions.

    Args:
        parameters: This dict contain the atom parameters.
        mixing: Determines how we should mix pair interactions.

    Returns:
        pair_param: A dictionary with the generated parameters.

    """
    atoms = []
    pair_param = {}
    # Get all atoms that appears as keys in the given
    # parameters:
    for key in parameters:
        if isinstance(key, tuple):
            pass
        atoms.append(key)

    # Loop over all pairs of atoms:
    for atmi, atmj in itertools.product(atoms, atoms):
        pari = parameters[atmi]
        parj = parameters[atmj]
        # Check if the pair is explicitly defined:
        if (atmi, atmj) in parameters:
            pair_param[atmi, atmj] = dict(parameters[atmi, atmj])
            pair_param[atmj, atmi] = pair_param[atmi, atmj]
            continue
        if (atmj, atmi) in parameters:
            pair_param[atmj, atmi] = dict(parameters[atmj, atmi])
            pair_param[atmi, atmj] = pair_param[atmj, atmi]
            continue
        if atmi == atmj:
            eps_ij = pari["epsilon"]
            sig_ij = pari["sigma"]
            rcut_ij = pari["rcut"]
            pair_param[atmi, atmi] = {
                "epsilon": eps_ij,
                "sigma": sig_ij,
                "rcut": rcut_ij,
            }
        else:
            eps_ij, sig_ij, rcut_ij = mix_parameters(
                epsilon_i=pari["epsilon"],
                sigma_i=pari["sigma"],
                rcut_i=pari["rcut"],
                epsilon_j=parj["epsilon"],
                sigma_j=parj["sigma"],
                rcut_j=parj["rcut"],
                mixing=mixing,
            )
            pair_param[atmi, atmj] = {
                "epsilon": eps_ij,
                "sigma": sig_ij,
                "rcut": rcut_ij,
            }
            # Save double as this in convenient:
            pair_param[atmj, atmi] = pair_param[atmi, atmj]
    return pair_param


def mix_parameters(
    epsilon_i: float,
    sigma_i: float,
    rcut_i: float,
    epsilon_j: float,
    sigma_j: float,
    rcut_j: float,
    mixing: str = "geometric",
) -> tuple[float, float, float]:
    r"""Calculate parameters according to the mixing rule.

    The available mixing rules are:

    1. Geometric:

       * .. math::

            \epsilon_{ij} = \sqrt{\epsilon_{i} \times \epsilon_{j}}

       * .. math::

            \sigma_{ij} = \sqrt{\sigma_{i} \times \sigma_{j}}

       * .. math::

            r_{\text{c},ij} = \sqrt{r_{\text{c},i} \times r_{\text{c},j}}

    2. Arithmetic:

       * .. math::

            \epsilon_{ij} = \sqrt{\epsilon_{i} \times \epsilon_{j}}

       * .. math::

            \sigma_{ij} = \frac{\sigma_{i} \times \sigma_{j}}{2}

       * .. math::

            r_{\text{c},ij} = \frac{r_{\text{c},i} \times r_{\text{c},j}}{2}

    3. Sixthpower

       * .. math::

            \epsilon_{ij} = 2 \sqrt{\epsilon_{i} \times \epsilon_{j}}
            \frac{\sigma_i^3 \times \sigma_j^3}{\sigma_i^6 + \sigma_j^6}

       * .. math::

            \sigma_{ij} = \left( \frac{\sigma_{i}^6 \times
            \sigma_{j}^6}{2} \right)^{1/6}

       * .. math::

            r_{\text{c},ij} = \left(\frac{r_{\text{c},i}^6 \times
            r_{\text{c},j}^6}{2}\right)^{1/6}

    Args:
        epsilon_i: Epsilon parameter for a particle of type `i`.
        sigma_i: Sigma parameter for a particle of type `i`.
        rcut_i: Cut-off value for a particle of type `i`.
        epsilon_j: Epsilon parameter for a particle of type `j`.
        sigma_j: Sigma parameter for a particle of type `j`.
        rcut_j: Cut-off value for a particle of type `j`.
        mixing: Represents what kind of mixing that should be done.

    Returns:
        out[0]: The mixed ``epsilon_ij`` parameter.
        out[1]: The mixed ``sigma_ij`` parameter.
        out[2]: The mixed ``rcut_ij`` parameter.

    """
    epsilon_ij = 0.0
    sigma_ij = 0.0
    rcut_ij = 0.0
    if mixing == "geometric":
        epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
        sigma_ij = np.sqrt(sigma_i * sigma_j)
        rcut_ij = np.sqrt(rcut_i * rcut_j)
    elif mixing == "arithmetic":
        epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        rcut_ij = 0.5 * (rcut_i + rcut_j)
    elif mixing == "sixthpower":
        si3 = sigma_i**3
        si6 = si3**2
        sj3 = sigma_j**3
        sj6 = sj3**2
        avgs6 = 0.5 * (si6 + sj6)
        epsilon_ij = np.sqrt(epsilon_i * epsilon_j) * si3 * sj3 / avgs6
        sigma_ij = avgs6 ** (1.0 / 6.0)
        rcut_ij = (0.5 * (rcut_i**6 + rcut_j**6)) ** (1.0 / 6.0)
    else:
        msg = f'Uknown mixing rule "{mixing}" requested!'
        raise ValueError(msg)
    return epsilon_ij, sigma_ij, rcut_ij
