import jax.numpy as jnp
import numpy as np
from jax import grad, jit

# float64 precision to avoid dihedral instability
from jax.config import config

from turtlemd.potentials.potential import Potential

config.update("jax_enable_x64", True)


@jit
def acosbound(x):
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))


@jit
def Vbond(x1, x2, k, r0):
    rij = x1 - x2
    rij_norm = jnp.linalg.norm(rij)
    V = 0.5 * k * (rij_norm - r0) ** 2
    return V


@jit
def Vangle(x1, x2, x3, k, theta0):
    rij = x1 - x2
    rkj = x3 - x2
    theta = acosbound(
        jnp.dot(rij, rkj) / (jnp.linalg.norm(rij) * jnp.linalg.norm(rkj))
    )
    V = 0.5 * k * (theta - theta0) ** 2
    return V


@jit
def Vdihedral(x1, x2, x3, x4, k, phi0, n):
    rij = x1 - x2
    rkj = x3 - x2
    rlk = x4 - x3
    t = jnp.cross(rij, rkj)
    u = jnp.cross(rlk, rkj)
    phi = acosbound(jnp.dot(t, u) / (jnp.linalg.norm(t) * jnp.linalg.norm(u)))
    V = k * (1.0 + jnp.cos(n * phi - phi0))
    return V


Fbond = jit(grad(Vbond, argnums=(0, 1)))
Fangle = jit(grad(Vangle, argnums=(0, 1, 2)))
Fdihedral = jit(grad(Vdihedral, argnums=(0, 1, 2, 3)))


class BondedInteractions(Potential):
    def __init__(
        self,
        bonds: list[tuple[float, float, int, int]],
        angles: list[tuple[float, float, int, int, int]],
        dihedrals: list[tuple[float, float, float, int, int, int, int]],
        desc="""Bond, angle and dihedral potential between particles
            with forces from automatic differentiation.""",
    ):
        """Initialise the bonded interactions."""
        super().__init__(desc=desc)
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def potential(self, system):
        pos = system.particles.pos
        pot = 0.0

        for bond in self.bonds:
            pot += Vbond(pos[bond[2]], pos[bond[3]], bond[0], bond[1])

        for angle in self.angles:
            pot += Vangle(
                pos[angle[2]],
                pos[angle[3]],
                pos[angle[4]],
                angle[0],
                angle[1],
            )

        for dihedral in self.dihedrals:
            pot += Vdihedral(
                pos[dihedral[3]],
                pos[dihedral[4]],
                pos[dihedral[5]],
                pos[dihedral[6]],
                dihedral[0],
                dihedral[1],
                dihedral[2],
            )

        return pot

    def force(self, system):
        pos = system.particles.pos
        force = np.zeros(pos.shape)
        for bond in self.bonds:
            f1, f2 = Fbond(pos[bond[2]], pos[bond[3]], bond[0], bond[1])
            force[bond[2], :] += -f1
            force[bond[3], :] += -f2

        for angle in self.angles:
            f1, f2, f3 = Fangle(
                pos[angle[2]],
                pos[angle[3]],
                pos[angle[4]],
                angle[0],
                angle[1],
            )
            force[angle[2], :] += -f1
            force[angle[3], :] += -f2
            force[angle[4], :] += -f3

        for dihedral in self.dihedrals:
            f1, f2, f3, f4 = Fdihedral(
                pos[dihedral[3]],
                pos[dihedral[4]],
                pos[dihedral[5]],
                pos[dihedral[6]],
                dihedral[0],
                dihedral[1],
                dihedral[2],
            )
            force[dihedral[3], :] += -jnp.nan_to_num(f1)
            force[dihedral[4], :] += -jnp.nan_to_num(f2)
            force[dihedral[5], :] += -jnp.nan_to_num(f3)
            force[dihedral[6], :] += -jnp.nan_to_num(f4)

        return force, 0.0
