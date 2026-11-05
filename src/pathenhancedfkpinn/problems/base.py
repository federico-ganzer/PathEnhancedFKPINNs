"""Problem definitions.

Parabolic:  d_t u = L u + F(x, u, grad_sigma u) + f(x),  u(., 0) = u0,
            L = b . grad + 1/2 tr(sigma sigma^T Hess),   grad_sigma u = sigma^T grad u.

Methods act on single points (x (d,), t scalar); vmap outside. Coefficients are jnp functions,
so exact derivatives for arm D come from autodiff of `exact`.

Sources f that do not depend on u go on the data side (offline); F goes on the model side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from pathenhancedfkpinn import derivs


@dataclass(frozen=True)
class Parabolic:
    """Base class. Subclasses set d, T, box, sigma and override b/f/u0/exact as needed."""

    d: int = 1
    T: float = 1.0
    box: tuple[tuple[float, float], ...] = ((-1.0, 1.0),)
    name: str = field(default="parabolic", compare=False)
    kind = "parabolic"
    linear = True  # F = 0

    # --- coefficients (single point) ---------------------------------------------------------

    def b(self, x: jax.Array) -> jax.Array:
        return jnp.zeros_like(x)

    def sigma(self) -> jax.Array:
        """Scalar, (d,) diagonal or (d, d)."""
        raise NotImplementedError

    def F(self, x, u, z):
        return jnp.zeros((), x.dtype)

    def f(self, x: jax.Array) -> jax.Array:
        return jnp.zeros((), x.dtype)

    def u0(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def exact(self, x: jax.Array, t) -> jax.Array:
        raise NotImplementedError

    def has_exact(self) -> bool:
        return type(self).exact is not Parabolic.exact

    def boundary_value(self, x, t):
        """Dirichlet data on the box faces: the exact solution (OPEN-8)."""
        return self.exact(x, t)

    # --- derived -----------------------------------------------------------------------------

    def diffusion_matrix(self, dtype=jnp.float64) -> jax.Array:
        """a = sigma sigma^T, (d, d)."""
        s = jnp.asarray(self.sigma(), dtype)
        if s.ndim == 0:
            return s**2 * jnp.eye(self.d, dtype=dtype)
        if s.ndim == 1:
            return jnp.diag(s**2)
        return s @ s.T

    def generator(self, g, x: jax.Array) -> jax.Array:
        """L g(x) = b . grad g + 1/2 tr(a Hess g) for a scalar function g of x."""
        a = self.diffusion_matrix(x.dtype)
        grad = jax.grad(g)(x)
        hess_term = jnp.sum(jax.vmap(lambda col: derivs.hvp(g, x, col))(a) * jnp.eye(self.d, dtype=x.dtype))
        return jnp.dot(self.b(x), grad) + 0.5 * hess_term

    def residual(self, u, x: jax.Array, t) -> jax.Array:
        """r = d_t u - L u - F - f for u(y, tau) a scalar function."""
        t = jnp.asarray(t, x.dtype)
        u_t = jax.grad(lambda tau: u(x, tau))(t)
        Lu = self.generator(lambda y: u(y, t), x)
        z = jnp.asarray(self.sigma(), x.dtype)
        gx = jax.grad(lambda y: u(y, t))(x)
        zs = gx @ z if z.ndim == 2 else z * gx
        return u_t - Lu - self.F(x, u(x, t), zs) - self.f(x)

    # --- sampling ----------------------------------------------------------------------------

    def lo_hi(self, dtype=jnp.float64):
        b = jnp.asarray(self.box, dtype)
        return b[:, 0], b[:, 1]

    def sample_interior(self, key, n: int):
        kx, kt = jax.random.split(key)
        lo, hi = self.lo_hi()
        x = lo + (hi - lo) * jax.random.uniform(kx, (n, self.d), jnp.float64)
        t = self.T * (1.0 - jax.random.uniform(kt, (n,), jnp.float64))  # t in (0, T]
        return x, t

    def sample_initial(self, key, n: int):
        lo, hi = self.lo_hi()
        x = lo + (hi - lo) * jax.random.uniform(key, (n, self.d), jnp.float64)
        return x, jnp.zeros((n,), jnp.float64)

    def sample_boundary(self, key, n: int):
        """Uniform on the box faces (face picked uniformly by axis and side), t ~ U(0, T]."""
        kx, ka, ks, kt = jax.random.split(key, 4)
        lo, hi = self.lo_hi()
        x = lo + (hi - lo) * jax.random.uniform(kx, (n, self.d), jnp.float64)
        axis = jax.random.randint(ka, (n,), 0, self.d)
        side = jax.random.bernoulli(ks, 0.5, (n,))
        face = jnp.where(side, hi[axis], lo[axis])
        x = x.at[jnp.arange(n), axis].set(face)
        t = self.T * (1.0 - jax.random.uniform(kt, (n,), jnp.float64))
        return x, t

    def lohi_with_time(self):
        lo, hi = zip(*self.box)
        return list(lo) + [0.0], list(hi) + [self.T]
