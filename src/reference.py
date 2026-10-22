"""Reference solutions for 2-D elliptic problems (elliptic.py): exact, or finite differences.

Both expose  ref(x) -> (N, 1)  and  ref.grad(x) -> (N, 2), for L2 and H1 errors.
Usually reached through `pde.reference(n)`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from losses.residual import gradient


class ExactReference:
    def __init__(self, pde) -> None:
        self.pde = pde

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.pde.exact(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            return gradient(self.pde.exact(x), x).detach()


class FDReference:
    """Finite differences for  a . u_xx + b . grad u - V u + f = 0,  a = sigma^2 / 2,
    on an n x n grid over the domain's bounding box.

    Unknowns are the grid nodes inside the domain. Nodes outside take g (staircase boundary:
    O(h) near curved or slanted boundaries). Reflecting walls use mirrored ghost nodes (zero
    flux). The drift uses exponential fitting, which is stable at any cell Peclet number
    |b| h / (2a) (`self.peclet`, the max) and second order where it is << 1; where it is large
    the scheme is effectively upwind (first order), so refine n if that region matters.

    Attributes for plotting: xs, ys (n,), U (n, n) in 'ij' order, inside (n, n).
    """

    def __init__(self, pde, n: int = 401) -> None:
        xs, ys, P, inside = pde.domain.grid(n, dtype=torch.float64)
        hx, hy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        inside = inside.numpy()
        ii, jj = np.nonzero(inside)
        Pin = P.reshape(n, n, 2)[ii, jj]  # (K, 2) interior nodes

        with torch.no_grad():
            U = pde.boundary_condition(P).reshape(n, n).numpy().copy()
            a = 0.5 * pde.diffusion(Pin).numpy() ** 2  # (K, 2)
            b = pde.drift(Pin).numpy()
            V = pde.potential(Pin).numpy()[:, 0]
            f = pde.source(Pin).numpy()[:, 0]

        K = ii.size
        number = -np.ones((n, n), dtype=np.int64)
        number[ii, jj] = np.arange(K)
        k = np.arange(K)

        # Il'in-Allen-Southwell exponential fitting: a -> a * Pe coth(Pe), Pe = b h / (2a).
        # Monotone (q stays in [min g, max g]) for any Pe; = central differences as Pe -> 0.
        Pe = b * np.array([hx, hy]) / (2 * a)
        small = np.abs(Pe) < 1e-6
        rho = np.where(small, 1.0 + Pe**2 / 3, Pe / np.tanh(np.where(small, 1.0, Pe)))
        self.peclet = float(np.abs(Pe).max())
        a = a * rho

        rows, cols = [k], [k]
        vals = [-2 * a[:, 0] / hx**2 - 2 * a[:, 1] / hy**2 - V]
        rhs = -f.copy()
        stencil = [
            (+1, 0, a[:, 0] / hx**2 + b[:, 0] / (2 * hx)),
            (-1, 0, a[:, 0] / hx**2 - b[:, 0] / (2 * hx)),
            (0, +1, a[:, 1] / hy**2 + b[:, 1] / (2 * hy)),
            (0, -1, a[:, 1] / hy**2 - b[:, 1] / (2 * hy)),
        ]
        for di, dj, coef in stencil:
            ni, nj = ii + di, jj + dj
            off = (ni < 0) | (ni >= n) | (nj < 0) | (nj >= n)
            if off.any():
                if not pde.domain.reflecting:
                    raise ValueError("interior node on the grid edge of a non-reflecting domain")
                ni = np.where(ni < 0, 1, np.where(ni >= n, n - 2, ni))  # mirror ghost node
                nj = np.where(nj < 0, 1, np.where(nj >= n, n - 2, nj))
            m = number[ni, nj]
            unknown = m >= 0
            rows.append(k[unknown])
            cols.append(m[unknown])
            vals.append(coef[unknown])
            rhs[~unknown] -= coef[~unknown] * U[ni[~unknown], nj[~unknown]]

        A = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(K, K)
        )
        U[ii, jj] = spsolve(A, rhs)

        self.xs, self.ys, self.U, self.inside = xs.numpy(), ys.numpy(), U, inside
        gx, gy = np.gradient(U, self.xs, self.ys)
        self._u = RegularGridInterpolator((self.xs, self.ys), U, bounds_error=False, fill_value=None)
        self._g = [
            RegularGridInterpolator((self.xs, self.ys), g, bounds_error=False, fill_value=None)
            for g in (gx, gy)
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        q = x.detach().cpu().double().numpy()
        return torch.as_tensor(self._u(q), dtype=x.dtype, device=x.device).unsqueeze(1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        q = x.detach().cpu().double().numpy()
        g = np.stack([gi(q) for gi in self._g], axis=1)
        return torch.as_tensor(g, dtype=x.dtype, device=x.device)
