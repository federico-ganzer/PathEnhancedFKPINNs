"""FK-PINN losses for stationary (elliptic) problems, after Tepakbong et al. 2026 (eq. 4.1-4.3).

    R_FK-PINN = lam_pde R_pde + lam_bc R_bc + lam_fk R_fk,
    R_fk      = mean_k (u(x_k) - u_hat_MC(x_k))^2     over a few FK supervision points.

Offline regime (paper Algorithm 2): FK labels are computed once, before training, with
`fk_exit_target`, so Monte-Carlo error is LABEL noise. A standard PINN is the same loss
without FK points.

    pts = sample_points(pde)                       # paper eq. E.7 split
    y_fk, Y, tau = fk_exit_target(pde, pts["x_fk"])  # stage 1
    loss, logs = fk_pinn_loss(model, pde, **pts, y_fk=y_fk, weights=w)   # stage 2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from losses.residual import gradient
from sde import fk_exit_paths


def fk_exit_target(
    pde,
    x: torch.Tensor,
    n_paths: int = 500,
    dt: float = 1e-3,
    t_max: float = math.inf,
    generator: torch.Generator | None = None,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monte-Carlo FK labels (paper eq. 3.2) at x (N, d).

    Returns (u_hat (N, 1), per-path samples Y (N, M), stopping times tau (N, M)). `tau` tells
    you what the labels cost. `chunk` bounds memory by simulating that many points at a time.
    """
    Ys, taus = [], []
    for xc in [x] if chunk is None else x.split(chunk):
        p = fk_exit_paths(pde, xc, n_paths, dt, t_max, generator)
        N, M = p["tau"].shape
        g = pde.boundary_condition(p["x_end"].reshape(N * M, -1)).reshape(N, M)
        Ys.append(p["run"] + p["disc"] * g)
        taus.append(p["tau"])
    Y = torch.cat(Ys)
    return Y.mean(dim=1, keepdim=True), Y, torch.cat(taus)


def sample_points(
    pde,
    n_coll: int = 10_000,
    n_bc: int = 400,
    p_data: float = 0.02,
    n_wall: int = 0,
    generator: torch.Generator | None = None,
    device=None,
) -> dict:
    """Paper eq. E.7: n_coll interior points; a random fraction p_data become FK points
    (x_fk), the rest residual points (x_int). Plus n_bc Dirichlet points (x_bc) and, for
    domains with reflecting walls, n_wall wall points with normals (walls)."""
    x = pde.sample_interior(n_coll, generator, device)
    n_fk = int(p_data * n_coll)
    perm = torch.randperm(n_coll, generator=generator).to(x.device)
    pts = {
        "x_int": x[perm[n_fk:]],
        "x_fk": x[perm[:n_fk]],
        "x_bc": pde.sample_boundary(n_bc, generator, device),
    }
    if n_wall:
        pts["walls"] = pde.domain.sample_walls(n_wall, generator, device)
    return pts


def residual_loss(model, pde, x: torch.Tensor) -> torch.Tensor:
    return pde.residual(model, x).pow(2).mean()


def dirichlet_loss(model, pde, x: torch.Tensor) -> torch.Tensor:
    return (model(x) - pde.boundary_condition(x)).pow(2).mean()


def neumann_loss(model, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Zero-flux walls: mean (n . grad u)^2."""
    x = x.detach().requires_grad_(True)
    return (gradient(model(x), x) * normals).sum(dim=1).pow(2).mean()


def data_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """FK supervision term R_fk (eq. 4.1)."""
    return (model(x) - y).pow(2).mean()


class UncertaintyWeights(nn.Module):
    """Trainable task weights (Kendall et al. 2018; paper sec. 4.2).

    total = sum_j exp(-s_j) R_j + s_j,  s_j = log sigma_j^2 clipped to `clip`, lam_j = exp(-s_j).
    Add its parameters to the optimiser. Initialise s_j so the initial weighted terms are of
    comparable size, e.g. init={"pde": log R_pde(0), ...}.
    """

    def __init__(self, names=("pde", "bc", "wall", "fk"), init: dict | None = None, clip=(-10.0, 10.0)):
        super().__init__()
        self.names = list(names)
        init = init or {}
        self.s = nn.Parameter(torch.tensor([float(init.get(n, 0.0)) for n in self.names]))
        self.clip = clip

    def forward(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        s = self.s.clamp(*self.clip)
        return sum(torch.exp(-s[i]) * terms[n] + s[i] for i, n in enumerate(self.names) if n in terms)

    def weights(self) -> dict[str, float]:
        s = self.s.detach().clamp(*self.clip)
        return {n: float(torch.exp(-s[i])) for i, n in enumerate(self.names)}


def fk_pinn_loss(
    model,
    pde,
    x_int: torch.Tensor,
    x_bc: torch.Tensor,
    x_fk: torch.Tensor | None = None,
    y_fk: torch.Tensor | None = None,
    walls: tuple[torch.Tensor, torch.Tensor] | None = None,
    weights: dict | UncertaintyWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq. 4.2. Terms: pde, bc, wall (if walls), fk (if x_fk). Returns (loss, per-term logs).

    weights: None (all 1), {name: float}, or an UncertaintyWeights module.
    Standard PINN: leave x_fk / y_fk out.
    """
    terms = {"pde": residual_loss(model, pde, x_int), "bc": dirichlet_loss(model, pde, x_bc)}
    if walls is not None:
        terms["wall"] = neumann_loss(model, *walls)
    if x_fk is not None:
        terms["fk"] = data_loss(model, x_fk, y_fk)

    if isinstance(weights, UncertaintyWeights):
        loss = weights(terms)
    else:
        w = weights or {}
        loss = sum(w.get(k, 1.0) * v for k, v in terms.items())
    return loss, {f"loss_{k}": float(v.detach()) for k, v in terms.items()}
