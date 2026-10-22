"""PDE specifications shared by the PINN (autodiff) and Feynman-Kac (SDE) routes.

Convention (parabolic, forward time), x in R^d, t in [t0, T], diagonal noise:

    du/dt = L u + f,    L u = 1/2 sum_i sigma_i^2 d^2u/dx_i^2 + b . grad u - V u
    u(x, t0) = u0(x)

Feynman-Kac with the SAME symbols. The SDE starts at x and runs for s = t - t0 on its own
clock r; PDE time runs backward as t - r:

    dX_r = b(X_r, t - r) dr + sigma(X_r, t - r) dW_r,    X_0 = x
    u(x, t) = E[ exp(-int_0^s V) u0(X_s) + int_0^s exp(-int_0^r V) f(X_r, t - r) dr ]

A new PDE only defines drift b, diffusion sigma, initial_condition (and optionally source f,
potential V, exact). The residual is derived from the same coefficients, so the PINN and the
FK-PINN cannot disagree about the problem.

Shapes: x (N, d), t (N, 1). drift/diffusion return (N, d); scalar fields return (N, 1).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from losses.residual import gradient


def _requires_grad(z: torch.Tensor) -> torch.Tensor:
    return z if z.requires_grad else z.detach().requires_grad_(True)


class PDE(ABC):
    def __init__(self, dim: int = 1, t0: float = 0.0, T: float = 1.0, domain=None) -> None:
        self.d = dim
        self.t0 = float(t0)
        self.T = float(T)
        if domain is None:
            domain = [[-1.0, 1.0]] * dim
        self.domain = torch.tensor(domain, dtype=torch.get_default_dtype())  # (d, 2)
        if self.domain.shape != (dim, 2):
            raise ValueError(f"domain must be {dim} [lo, hi] pairs, got {list(self.domain.shape)}")

    # -- coefficients --------------------------------------------------------------------

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """b(x, t), shape (N, d)."""

    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Diagonal sigma(x, t), shape (N, d)."""

    def source(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t), shape (N, 1)."""
        return x.new_zeros(x.shape[0], 1)

    def potential(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """V(x, t) >= 0, shape (N, 1). Reaction / discount term."""
        return x.new_zeros(x.shape[0], 1)

    @property
    def inhomogeneous(self) -> bool:
        """True if source or potential is overridden, so FK must accumulate running terms."""
        cls = type(self)
        return cls.source is not PDE.source or cls.potential is not PDE.potential

    # -- conditions and truth ------------------------------------------------------------

    @abstractmethod
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """u0(x), shape (N, 1)."""

    def exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
        """Analytic solution if known, else None."""
        return None

    def boundary_condition(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Dirichlet data g(x, t). Defaults to `exact` when known, else zero."""
        u = self.exact(x, t)
        return u if u is not None else x.new_zeros(x.shape[0], 1)

    # -- AD residual ---------------------------------------------------------------------

    def residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """du/dt - L u - f at (x, t), shape (N, 1). model(cat([x, t])) -> (N, 1)."""
        x, t = _requires_grad(x), _requires_grad(t)
        u = model(torch.cat([x, t], dim=1))
        u_t = gradient(u, t)
        u_x = gradient(u, x)  # (N, d)
        u_xx = torch.cat([gradient(u_x[:, i : i + 1], x)[:, i : i + 1] for i in range(self.d)], 1)
        sigma = self.diffusion(x, t)
        Lu = (
            0.5 * (sigma**2 * u_xx).sum(dim=1, keepdim=True)
            + (self.drift(x, t) * u_x).sum(dim=1, keepdim=True)
            - self.potential(x, t) * u
        )
        return u_t - Lu - self.source(x, t)

    # -- sampling (box domain) -----------------------------------------------------------

    def sample_interior(self, n: int, device=None, generator=None):
        """Uniform (x, t) in domain x [t0, T]."""
        lo, hi = self.domain[:, 0].to(device), self.domain[:, 1].to(device)
        x = torch.rand(n, self.d, device=device, generator=generator) * (hi - lo) + lo
        t = torch.rand(n, 1, device=device, generator=generator) * (self.T - self.t0) + self.t0
        return x, t

    def sample_initial(self, n: int, device=None, generator=None):
        lo, hi = self.domain[:, 0].to(device), self.domain[:, 1].to(device)
        x = torch.rand(n, self.d, device=device, generator=generator) * (hi - lo) + lo
        return x, torch.full((n, 1), self.t0, device=device)

    def sample_boundary(self, n: int, device=None, generator=None):
        """Uniform on the boundary of the box: pick a face, then a point on it."""
        lo, hi = self.domain[:, 0].to(device), self.domain[:, 1].to(device)
        x = torch.rand(n, self.d, device=device, generator=generator) * (hi - lo) + lo
        axis = torch.randint(0, self.d, (n,), device=device, generator=generator)
        side = torch.randint(0, 2, (n,), device=device, generator=generator)
        x[torch.arange(n, device=device), axis] = torch.where(side.bool(), hi[axis], lo[axis])
        t = torch.rand(n, 1, device=device, generator=generator) * (self.T - self.t0) + self.t0
        return x, t


class Heat(PDE):
    """du/dt = nu * lap(u).   FK: b = 0, sigma = sqrt(2 nu)  (sigma^2 / 2 = nu, not sigma = nu).

    u0 = prod_i sin(k pi x_i)  ->  u = exp(-nu d (k pi)^2 (t - t0)) u0(x).
    For integer k this is zero on the boundary of [-1, 1]^d, so the free-space FK solution
    also solves the homogeneous Dirichlet problem on the box.
    """

    def __init__(self, dim: int = 1, nu: float = 0.1, k: float = 1.0, **kw) -> None:
        super().__init__(dim, **kw)
        self.nu = float(nu)
        self.k = float(k)

    def drift(self, x, t):
        return torch.zeros_like(x)

    def diffusion(self, x, t):
        return torch.full_like(x, math.sqrt(2.0 * self.nu))

    def initial_condition(self, x):
        return torch.sin(self.k * math.pi * x).prod(dim=1, keepdim=True)

    def exact(self, x, t):
        decay = torch.exp(-self.nu * self.d * (self.k * math.pi) ** 2 * (t - self.t0))
        return decay * self.initial_condition(x)


class ADR(Heat):
    """Advection-diffusion-reaction: du/dt = nu lap(u) - c . grad(u) - r u.

    FK: b = -c, sigma = sqrt(2 nu), V = r. With u0 = prod_i sin(k pi x_i) the whole-space
    solution is a decaying travelling wave (tau = t - t0):

        u = exp(-(nu d (k pi)^2 + r) tau) prod_i sin(k pi (x_i - c_i tau))

    It does not vanish on the box, so boundary data comes from `exact` (base-class default).
    """

    def __init__(self, dim: int = 1, nu: float = 0.1, c=1.0, r: float = 0.0, k: float = 1.0, **kw):
        super().__init__(dim, nu=nu, k=k, **kw)
        c = torch.as_tensor(c, dtype=torch.get_default_dtype())
        self.c = c.expand(dim).clone() if c.ndim == 0 else c.clone()  # (d,)
        if self.c.shape != (dim,):
            raise ValueError(f"c must be a scalar or have {dim} entries")
        self.r = float(r)

    def drift(self, x, t):
        return torch.zeros_like(x) - self.c.to(x)

    def potential(self, x, t):
        return x.new_full((x.shape[0], 1), self.r)

    def exact(self, x, t):
        tau = t - self.t0
        lam = self.nu * self.d * (self.k * math.pi) ** 2 + self.r
        wave = torch.sin(self.k * math.pi * (x - self.c.to(x) * tau)).prod(dim=1, keepdim=True)
        return torch.exp(-lam * tau) * wave
