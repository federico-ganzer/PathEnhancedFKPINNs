"""Standard AD-based PINN loss and the IC/BC terms shared by every method.

    L = w_r * MSE(residual) + w_ic * MSE(u - u0) + w_bc * MSE(u - g)

Models take a single input `cat([x, t], dim=1)` of shape (N, d+1) and return (N, 1).
"""

from __future__ import annotations

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx with create_graph=True. y (N, 1), x (N, d) -> (N, d)."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """sum_i d^2y/dx_i^2, shape (N, 1). Exact: d extra backward passes."""
    gy = gradient(y, x)
    out = torch.zeros_like(y)
    for i in range(x.shape[1]):
        out = out + gradient(gy[:, i : i + 1], x)[:, i : i + 1]
    return out


def initial_loss(model, pde, n: int, generator=None, device=None) -> torch.Tensor:
    x, t = pde.sample_initial(n, device=device, generator=generator)
    return (model(torch.cat([x, t], dim=1)) - pde.initial_condition(x)).pow(2).mean()


def boundary_loss(model, pde, n: int, generator=None, device=None) -> torch.Tensor:
    x, t = pde.sample_boundary(n, device=device, generator=generator)
    return (model(torch.cat([x, t], dim=1)) - pde.boundary_condition(x, t)).pow(2).mean()


def hard_constraint(model, pde):
    """Ansatz u(x,t) = u0(x) + (t - t0) * B(x) * N(x,t), B = prod_i (x_i - lo_i)(hi_i - x_i).

    Exact at t = t0, and exact on the boundary only if g(x,t) = u0(x) there (true for the
    homogeneous-Dirichlet heat mode, not in general). Returns a function u(x, t).
    """
    lo, hi = pde.domain[:, 0], pde.domain[:, 1]

    def u(x, t):
        b = ((x - lo.to(x)) * (hi.to(x) - x)).prod(dim=1, keepdim=True)
        return pde.initial_condition(x) + (t - pde.t0) * b * model(torch.cat([x, t], dim=1))

    return u


def pinn_loss(
    model,
    pde,
    n_interior: int = 1024,
    n_initial: int = 256,
    n_boundary: int = 256,
    w_residual: float = 1.0,
    w_initial: float = 1.0,
    w_boundary: float = 1.0,
    generator: torch.Generator | None = None,
    device=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total loss, per-term logs). Fresh collocation points every call."""
    xr, tr = pde.sample_interior(n_interior, device=device, generator=generator)
    l_res = pde.residual(model, xr, tr).pow(2).mean()
    l_ic = initial_loss(model, pde, n_initial, generator=generator, device=device)
    l_bc = boundary_loss(model, pde, n_boundary, generator=generator, device=device)

    loss = w_residual * l_res + w_initial * l_ic + w_boundary * l_bc
    logs = {
        "loss_residual": float(l_res.detach()),
        "loss_initial": float(l_ic.detach()),
        "loss_boundary": float(l_bc.detach()),
    }
    return loss, logs
