"""Ornstein-Uhlenbeck benchmark.

dX = -a X dt + sqrt(2 nu) dW,  d_t u = -a x . grad u + nu Lap u,  u0 = prod_i sin(pi x_i).
Defaults: a = 1, nu = 0.05, T in {0.25, 1}, box [-1.5, 1.5]^d, exact boundary values on the faces.

Mehler: u = prod_i exp(-pi^2 s^2 / 2) sin(pi x_i e^{-a t}),  s^2 = nu (1 - e^{-2 a t}) / a.
R_loc weight tau(t) = nu (e^{2 a t} - 1) / a (draft eq. 12, needs N0 = infinity).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from pathenhancedfkpinn.problems.base import Parabolic


@dataclass(frozen=True)
class OU(Parabolic):
    a: float = 1.0
    nu: float = 0.05
    box: tuple[tuple[float, float], ...] = ((-1.5, 1.5),)
    name: str = "ou"

    def b(self, x):
        return -self.a * x

    def sigma(self):
        return jnp.sqrt(2.0 * self.nu)

    def u0(self, x):
        return jnp.prod(jnp.sin(jnp.pi * x))

    def exact(self, x, t):
        '''FK solution (Mehler formula) for the OU process.'''
        s2 = self.nu * (1.0 - jnp.exp(-2.0 * self.a * t)) / self.a
        return jnp.prod(jnp.exp(-(jnp.pi**2) * s2 / 2.0) * jnp.sin(jnp.pi * x * jnp.exp(-self.a * t)))

    def tau(self, t):
        return self.nu * (jnp.exp(2.0 * self.a * t) - 1.0) / self.a


def ou_box(d: int, half_width: float = 1.5) -> tuple[tuple[float, float], ...]:
    return tuple((-half_width, half_width) for _ in range(d))
