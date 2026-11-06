"""Manufactured parabolic benchmark.

u*(x) = sin(pi x) + eps sin(k pi x),  x in [-1, 1],  d = 1,  nu = 0.05.

variant "static_source" (default reading of OPEN-5):
    d_t u = nu u_xx + f,  f = -nu u*'',  u0 = u*  ->  u = u* for all t.
    FK payoff Y = u0(X_t) + int_0^t f(X_s) ds (source on the data side).
variant "two_mode_heat" (alternative):
    d_t u = nu u_xx,  u0 = u*  ->  u = e^{-nu pi^2 t} sin(pi x) + eps e^{-nu (k pi)^2 t} sin(k pi x).

Both vanish at x = +-1 for integer k. Spectral predictions use lambda_max = nu (k pi)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from pathenhancedfkpinn.problems.base import Parabolic


@dataclass(frozen=True)
class Manufactured(Parabolic):
    nu: float = 0.05
    k: int = 4
    eps: float = 0.1
    variant: str = "static_source"
    name: str = "manufactured"

    def __post_init__(self):
        if self.variant not in ("static_source", "two_mode_heat"):
            raise ValueError(f"unknown variant {self.variant!r}")

    def sigma(self):
        return jnp.sqrt(2.0 * self.nu)

    def _ustar(self, x):
        x = x[0]
        return jnp.sin(jnp.pi * x) + self.eps * jnp.sin(self.k * jnp.pi * x)

    def u0(self, x):
        return self._ustar(x)

    def f(self, x):
        if self.variant == "two_mode_heat":
            return jnp.zeros((), x.dtype)
        x = x[0]
        k = self.k
        return self.nu * jnp.pi**2 * (jnp.sin(jnp.pi * x) + self.eps * k**2 * jnp.sin(k * jnp.pi * x))

    def exact(self, x, t):
        if self.variant == "static_source":
            return self._ustar(x) + 0.0 * t
        x = x[0]
        k = self.k
        return jnp.exp(-self.nu * jnp.pi**2 * t) * jnp.sin(jnp.pi * x) + self.eps * jnp.exp(
            -self.nu * (k * jnp.pi) ** 2 * t
        ) * jnp.sin(k * jnp.pi * x)

    @property
    def lambda_max(self) -> float:
        return float(self.nu * (self.k * jnp.pi) ** 2)

    def tau(self, t):
        return 2.0 * self.nu * t
