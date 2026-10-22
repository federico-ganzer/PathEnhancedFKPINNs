"""Feynman-Kac loss: regress the network onto a Monte-Carlo FK estimate at the same point.

    L(theta) = E_(x,t) [ (u_theta(x,t) - Y_hat(x,t))^2 ]
    Y_hat    = (1/M) sum_m [ exp(-int V) u0(X^m_end) + int exp(-int V) f dr ]

`horizon=None` integrates all the way to t0 (unbiased). A float h bootstraps off the
network at t - h (cheaper, biased while the network is wrong).
"""

from __future__ import annotations

import torch
from torch import func

from metrics import mc_summary
from sde import fk_paths


def fk_target(
    pde,
    x: torch.Tensor,
    t: torch.Tensor,
    n_paths: int = 256,
    n_steps: int = 50,
    antithetic: bool = False,
    model: torch.nn.Module | None = None,
    horizon: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo FK estimate at (x, t). Returns (target (N, 1), per-path samples Y (N, M))."""
    N, d = x.shape
    M = n_paths
    if horizon is None:
        t_end = torch.full_like(t, pde.t0)
    else:
        if model is None:
            raise ValueError("horizon bootstrapping needs a model")
        t_end = (t - horizon).clamp_min(pde.t0)

    disc = torch.ones(N, M, 1, dtype=x.dtype, device=x.device)  # exp(-int_0^r V)
    acc = torch.zeros_like(disc)  # int_0^r exp(-int V) f dr'

    def accumulate(X, s, dt):
        nonlocal acc, disc
        f = pde.source(X, s).reshape(N, M, 1)
        V = pde.potential(X, s).reshape(N, M, 1)
        acc = acc + disc * f * dt  # left Riemann, before this step's V is folded in
        disc = disc * torch.exp(-V * dt)

    paths, _ = fk_paths(
        pde,
        x,
        t,
        n_paths=M,
        n_steps=n_steps,
        t_end=t_end,
        generator=generator,
        antithetic=antithetic,
        return_paths=False,
        callback=accumulate if pde.inhomogeneous else None,
    )
    Xf = paths[:, :, -1, :].reshape(N * M, d)

    if horizon is None:
        g_end = pde.initial_condition(Xf).reshape(N, M, 1)
    else:
        te = t_end.reshape(N, 1, 1).expand(N, M, 1).reshape(N * M, 1)
        g_end = model(torch.cat([Xf, te], dim=1)).reshape(N, M, 1)

    Y = (disc * g_end + acc).squeeze(-1)
    return Y.mean(dim=1, keepdim=True), Y


def fk_loss(
    model: torch.nn.Module,
    pde,
    n_points: int = 256,
    n_paths: int = 256,
    n_steps: int = 50,
    antithetic: bool = False,
    horizon: float | None = None,
    detach_target: bool = True,
    generator: torch.Generator | None = None,
    device=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (loss, logs). Logs include the MC estimator's variance and standard error."""
    x, t = pde.sample_interior(n_points, device=device, generator=generator)
    target, Y = fk_target(
        pde, x, t, n_paths, n_steps, antithetic, model=model, horizon=horizon, generator=generator
    )
    if detach_target:
        target = target.detach()
    u = model(torch.cat([x, t], dim=1))
    loss = (u - target).pow(2).mean()
    logs = {"fk_loss": float(loss.detach()), **mc_summary(Y), "n_paths": float(n_paths)}
    return loss, logs


def per_sample_grads(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """J[i] = grad_theta u_theta(x_i, t_i), flattened over parameters. Shape (B, P)."""
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def u_one(p, xt):
        return func.functional_call(model, (p, buffers), (xt.unsqueeze(0),)).squeeze()

    gs = func.vmap(func.grad(u_one), in_dims=(None, 0))(params, torch.cat([x, t], dim=1))
    return torch.cat([v.reshape(x.shape[0], -1) for v in gs.values()], dim=1)


def kappa_hat(
    model: torch.nn.Module,
    pde,
    n_points: int = 256,
    n_paths: int = 256,
    n_steps: int = 50,
    antithetic: bool = False,
    m: int = 2,
    split: bool = False,
    generator: torch.Generator | None = None,
    device=None,
) -> dict[str, float]:
    """Gradient-noise split of the FK-PINN loss (Prop A.10). Returns dict(A, H, S2, kappa, H_neg).

    A = MC (target) part, H = collocation part, kappa = H_+ / (A + H_+).
    split=False: m independent targets of M paths each -> kappa at the training M (m x cost).
    split=True : one batch of M split into m groups -> free, A rescaled by 1/m to express it at M.
    H < 0 late in training means "indistinguishable from zero"; report it, don't hide it.
    """
    x, t = pde.sample_interior(n_points, device=device, generator=generator)
    B = n_points
    with torch.no_grad():
        u = model(torch.cat([x, t], dim=1)).squeeze(1)  # (B,)

        if split:
            _, Y = fk_target(pde, x, t, n_paths, n_steps, antithetic, generator=generator)
            Ys = Y.reshape(B, m, -1).mean(-1)  # (B, m), each of M/m paths
        else:
            Ys = torch.stack(
                [
                    fk_target(pde, x, t, n_paths, n_steps, antithetic, generator=generator)[0]
                    .squeeze(1)
                    for _ in range(m)
                ],
                dim=1,
            )  # (B, m), each of M paths

    J = per_sample_grads(model, x, t)  # (B, P)
    gn2 = (J**2).sum(1)  # ||g_i||^2

    s2 = Ys.var(dim=1, unbiased=True)
    if split:
        s2 = s2 / m  # variance of an M-path target
    A = 4.0 * (s2 * gn2).mean()

    c = 2.0 * (u - Ys.mean(1))  # (B,)
    Gbar = c[:, None] * J  # (B, P)
    S2 = (Gbar - Gbar.mean(0)).pow(2).sum(1).sum() / (B - 1)  # tr Cov(Gbar) = H + A/m

    H = S2 - A / m
    Hp = H.clamp_min(0.0)
    return {
        "A": float(A),
        "H": float(H),
        "S2": float(S2),
        "kappa": float(Hp / (A + Hp + 1e-300)),
        "H_neg": bool(H < 0),
    }
