"""Heat equation d_t u = nu Lap u. FK: b = 0, sigma = sqrt(2 nu).

u0 = prod_i sin(k pi x_i)  ->  u = exp(-nu d (k pi)^2 t) u0(x).
For integer k, u vanishes on the faces of [-1, 1]^d, so FK paths on R^d solve the Dirichlet
problem on the box exactly (oddness about the faces).

Signature coefficients (d = 1): beta_{1^n} = (sigma k pi)^n exp(-nu (k pi)^2 t) sin(k pi x + n pi/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from pathenhancedfkpinn.problems.base import Parabolic


@dataclass(frozen=True)
class Heat(Parabolic):
    nu: float = 0.05
    k: float = 1.0
    name: str = "heat"

    def sigma(self):
        return jnp.sqrt(2.0 * self.nu)

    def u0(self, x):
        return jnp.prod(jnp.sin(self.k * jnp.pi * x))

    def exact(self, x, t):
        '''FK solution for the heat equation with Dirichlet boundary conditions.'''
        return jnp.exp(-self.nu * self.d * (self.k * jnp.pi) ** 2 * t) * self.u0(x)

    def tau(self, t):
        """tau(t) = 2 nu t in R_loc (draft eq. 12)."""
        return 2.0 * self.nu * t
