"""Stationary (elliptic) Dirichlet problems with exit-time Feynman-Kac representations, and the
four benchmarks of Tepakbong et al. 2026 ("Taming the loss landscape of PINNs with noisy
Feynman-Kac supervision", to_read/2606.00643v2.pdf).

Convention (paper App. A, eqs. A.5-A.6), diagonal noise:

    L u - V u + f = 0   in D,        u = g   on the Dirichlet boundary,
    L u = b . grad u + 1/2 sum_i sigma_i^2 d^2u/dx_i^2

    u(x) = E_x[ int_0^tau e^{-int_0^s V} f(X_s) ds + e^{-int_0^tau V} g(X_tau) ],
    dX = b(X) dt + sigma(X) dW,    tau = first exit time from D.

Notation clash: the paper calls the Schrodinger potential V and the energy landscapes V too.
Here V is always the FK killing term (`potential`); energy landscapes are U.

Models take x (N, d) only -- no time input. Coefficients take x (N, d).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from domains import Box, BoxMinusDisks, Hexagon
from losses.residual import gradient

# Training / Monte-Carlo settings of the paper (App. E.2.1). n_coll / n_bc are stated for
# Poisson only; the paper does not give them for the other three problems.
PAPER = {
    "width": 128,
    "depth": 4,
    "activation": "tanh",
    "adam_steps": 30_000,  # lr 1e-3 -> 1e-6 by step-wise exponential decay
    "lbfgs_steps": 15_000,  # then L-BFGS fine-tuning
    "lr": 1e-3,
    "lr_final": 1e-6,
    "dt": 1e-3,  # Euler-Maruyama step for FK labels
    "n_mc": 500,  # paths per FK point
    "t_max": math.inf,  # simulate until exit
    "p_data": 0.02,  # fraction of collocation points that get an FK label (eq. E.7)
    "n_coll": 10_000,
    "n_bc": 400,
    "n_seeds": 5,
}


def _requires_grad(z: torch.Tensor) -> torch.Tensor:
    return z if z.requires_grad else z.detach().requires_grad_(True)


class EllipticPDE(ABC):
    def __init__(self, domain) -> None:
        self.domain = domain
        self.d = domain.d

    # -- coefficients --------------------------------------------------------------------

    @abstractmethod
    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """b(x), shape (N, d)."""

    @abstractmethod
    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """Diagonal sigma(x), shape (N, d)."""

    def source(self, x: torch.Tensor) -> torch.Tensor:
        """f(x), shape (N, 1)."""
        return x.new_zeros(x.shape[0], 1)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """V(x), shape (N, 1). FK killing / discount term."""
        return x.new_zeros(x.shape[0], 1)

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """g(x), shape (N, 1). Also evaluated at the (slightly outside) exit points of FK paths."""
        return x.new_zeros(x.shape[0], 1)

    def exact(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    @property
    def inhomogeneous(self) -> bool:
        """True if source or potential is overridden, so FK must accumulate running terms."""
        cls = type(self)
        return cls.source is not EllipticPDE.source or cls.potential is not EllipticPDE.potential

    # -- AD residual ---------------------------------------------------------------------

    def residual(self, model, x: torch.Tensor) -> torch.Tensor:
        """L u - V u + f at x, shape (N, 1)."""
        x = _requires_grad(x)
        u = model(x)
        u_x = gradient(u, x)
        u_xx = torch.cat([gradient(u_x[:, i : i + 1], x)[:, i : i + 1] for i in range(self.d)], 1)
        sigma = self.diffusion(x)
        Lu = 0.5 * (sigma**2 * u_xx).sum(dim=1, keepdim=True) + (self.drift(x) * u_x).sum(
            dim=1, keepdim=True
        )
        return Lu - self.potential(x) * u + self.source(x)

    # -- sampling and reference ----------------------------------------------------------

    def sample_interior(self, n: int, generator=None, device=None) -> torch.Tensor:
        return self.domain.sample_interior(n, generator, device)

    def sample_boundary(self, n: int, generator=None, device=None) -> torch.Tensor:
        return self.domain.sample_boundary(n, generator, device)

    def reference(self, n: int = 401):
        """Exact solution if known, else a finite-difference solve on an n x n grid.

        Returns an object with __call__(x) -> (N, 1) and grad(x) -> (N, d).
        """
        from reference import ExactReference, FDReference

        if type(self).exact is not EllipticPDE.exact:
            return ExactReference(self)
        return FDReference(self, n)


# --------------------------------------------------------------------------------------
# Energy landscapes U and their gradients (closed form, cheap inside the SDE loop)
# --------------------------------------------------------------------------------------


def double_well(x: torch.Tensor, alpha: float = 1.0):
    """U = 1/4 (x1^2 - 1)^2 + alpha/2 x2^2. Returns (U (N,1), grad U (N,2))."""
    x1, x2 = x[:, :1], x[:, 1:2]
    U = 0.25 * (x1**2 - 1) ** 2 + 0.5 * alpha * x2**2
    return U, torch.cat([x1 * (x1**2 - 1), alpha * x2], dim=1)


# Paper Table 3.
MULLER_BROWN = {
    "A": [-200.0, -100.0, -170.0, 15.0],
    "a": [-1.0, -1.0, -6.5, 0.7],
    "b": [0.0, 0.0, 11.0, 0.6],
    "c": [-10.0, -10.0, -6.5, 0.7],
    "x0": [1.0, 0.0, -0.5, -1.0],
    "y0": [0.0, 0.5, 1.5, 1.0],
}


def muller_brown(x: torch.Tensor):
    """U = sum_k A_k exp(a_k dx^2 + b_k dx dy + c_k dy^2) (paper eq. E.15). Returns (U, grad U)."""
    p = {k: torch.tensor(v, dtype=x.dtype, device=x.device) for k, v in MULLER_BROWN.items()}
    dx, dy = x[:, :1] - p["x0"], x[:, 1:2] - p["y0"]  # (N, 4)
    e = p["A"] * torch.exp(p["a"] * dx**2 + p["b"] * dx * dy + p["c"] * dy**2)
    gx = (e * (2 * p["a"] * dx + p["b"] * dy)).sum(dim=1, keepdim=True)
    gy = (e * (p["b"] * dx + 2 * p["c"] * dy)).sum(dim=1, keepdim=True)
    return e.sum(dim=1, keepdim=True), torch.cat([gx, gy], dim=1)


# --------------------------------------------------------------------------------------
# The four benchmark problems
# --------------------------------------------------------------------------------------


class Poisson(EllipticPDE):
    """-lap u = f on [0,1]^2, u = 0 on the boundary (paper E.2.2).

    u* = sum_m A_m sin(k_m pi x1) sin(l_m pi x2), modes (A, k, l) = (3,1,4), (6,5,1), (3,2,2),
    (4,5,6), f = pi^2 sum_m A_m (k_m^2 + l_m^2) sin sin. FK: b = 0, sigma = sqrt(2), f, g = 0.
    """

    MODES = ((3.0, 1, 4), (6.0, 5, 1), (3.0, 2, 2), (4.0, 5, 6))

    def __init__(self, modes=MODES) -> None:
        super().__init__(Box([[0.0, 1.0], [0.0, 1.0]]))
        self.modes = tuple(modes)

    def drift(self, x):
        return torch.zeros_like(x)

    def diffusion(self, x):
        return torch.full_like(x, math.sqrt(2.0))

    def _sum(self, x, weight):
        out = x.new_zeros(x.shape[0], 1)
        for A, k, l in self.modes:
            s = torch.sin(k * math.pi * x[:, :1]) * torch.sin(l * math.pi * x[:, 1:2])
            out = out + weight(A, k, l) * s
        return out

    def exact(self, x):
        return self._sum(x, lambda A, k, l: A)

    def source(self, x):
        return self._sum(x, lambda A, k, l: math.pi**2 * A * (k**2 + l**2))


class Schrodinger(EllipticPDE):
    """-1/2 lap psi + V psi = f on [-1,1]^2, psi = 0 on the boundary (paper eq. 7.1-7.3).

    V = -V0 (cos k x1 + cos k x2), k = 2 pi / a (crystal lattice); f = A sin(K x1) sin(K x2).
    Paper: V0 = 5, a = 0.5, A = 50, K = 5. FK: b = 0, sigma = 1, killing V (negative in places,
    so the discount e^{-int V} can exceed 1), source f, g = 0. No closed form (paper uses FEM).
    """

    def __init__(self, V0=5.0, a=0.5, A=50.0, K=5.0) -> None:
        super().__init__(Box([[-1.0, 1.0], [-1.0, 1.0]]))
        self.V0, self.k, self.A, self.K = float(V0), 2 * math.pi / float(a), float(A), float(K)

    def drift(self, x):
        return torch.zeros_like(x)

    def diffusion(self, x):
        return torch.ones_like(x)

    def potential(self, x):
        return -self.V0 * torch.cos(self.k * x).sum(dim=1, keepdim=True)

    def source(self, x):
        return self.A * torch.sin(self.K * x).prod(dim=1, keepdim=True)


class MeanExitTime(EllipticPDE):
    """Mean first exit time from a hexagon (paper E.1.2, E.2.3).

        beta^{-1} lap tau - grad U . grad tau = -1   in the regular hexagon of circumradius R,
        tau = 0 on the boundary,   U = double well 1/4 (x1^2-1)^2 + alpha/2 x2^2.
    Paper: R = 2, alpha = 1, beta = 5. FK: b = -grad U, sigma = sqrt(2/beta), f = 1, g = 0, so
    tau(x) = E_x[exit time]. Values reach ~20, so each path costs ~2e4 steps at dt = 1e-3.
    Discrete exit monitoring biases the labels upward (paper Prop. 6.1: C sqrt(dt)), and the
    steep boundary layer makes C large here. Measured against the FD reference (~20.0 at x = 0):
    dt = 1e-2 -> ~25, dt = 1e-3 -> 21.7 +- 0.5, dt = 2.5e-4 -> 21.4 +- 0.5 (M = 2000).
    The paper's FK labels (dt = 1e-3) therefore carry a ~5-8% bias (cf. its Fig. 13a).
    """

    def __init__(self, beta=5.0, alpha=1.0, R=2.0) -> None:
        super().__init__(Hexagon(R))
        self.beta, self.alpha = float(beta), float(alpha)

    def energy(self, x):
        return double_well(x, self.alpha)[0]

    def drift(self, x):
        return -double_well(x, self.alpha)[1]

    def diffusion(self, x):
        return torch.full_like(x, math.sqrt(2.0 / self.beta))

    def source(self, x):
        return x.new_ones(x.shape[0], 1)


class Committor(EllipticPDE):
    """Committor of overdamped Langevin dynamics in the Muller-Brown potential (E.1.3, E.2.4).

        beta^{-1} lap q - grad U . grad q = 0   in box minus (A u B),
        q = 0 on dA,  q = 1 on dB,
    box [-1.5, 1.2] x [-0.2, 2], A and B disks of radius 0.2 at the reactant / product minima.
    FK: b = -grad U, sigma = sqrt(2/beta), f = 0, g = 1 on B -> q(x) = P_x(hit B before A).

    NOT specified in the paper (assumptions -- change if you learn otherwise):
      beta   default 0.2 (kT = 5): fitted by matching our FD reference to the colours of the
             paper's Fig. 8 reference (RMSE 0.023 over ~33k pixels; 0.1 gives 0.070, 0.5 gives
             0.055). Probably the paper's beta = 5 used as a temperature. Taken literally as an
             inverse temperature, beta = 5 freezes the dynamics (barriers ~100) and q is a step.
      walls  box edges are reflecting (zero flux): FK paths are folded back, the PINN needs a
             Neumann term on `domain.sample_walls`, and the FD reference mirrors ghost nodes.
    dt     |grad U| > 100 on most of the box (~1e4 at the top-right corner). At beta = 0.2,
           FK labels at the paper's dt = 1e-3 are biased by ~+0.02 (3 SE at M = 500);
           dt = 1e-4 is unbiased and still cheap (paths exit in ~0.03 time units).
    """

    def __init__(
        self,
        beta=0.2,
        radius=0.2,
        center_A=(-0.5582, 1.4417),
        center_B=(0.6235, 0.0281),
        bounds=((-1.5, 1.2), (-0.2, 2.0)),
    ) -> None:
        super().__init__(BoxMinusDisks(bounds, [center_A, center_B], [radius, radius]))
        self.beta = float(beta)

    def energy(self, x):
        return muller_brown(x)[0]

    def drift(self, x):
        return -muller_brown(x)[1]

    def diffusion(self, x):
        return torch.full_like(x, math.sqrt(2.0 / self.beta))

    def boundary_condition(self, x):
        """1 at B, 0 at A (nearest disk decides, so exit points just inside a disk work)."""
        return (self.domain.nearest_disk(x) == 1).to(x.dtype).unsqueeze(1)


PROBLEMS = {
    "poisson": Poisson,
    "schrodinger": Schrodinger,
    "mean_exit_time": MeanExitTime,
    "committor": Committor,
}
