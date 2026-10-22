"""SDE integration and Feynman-Kac path generation.

    dX_r = b(X_r, r) dr + sigma(X_r, r) dW_r,    sigma diagonal

Shapes: x0 (N, d) start points; paths (N, M, K+1, d) for M paths per start and K steps,
endpoints included. Inside the integrator, drift/diffusion see x (N, M, d) and t (N, 1, 1).

Parabolic PDEs (pdes.py): `fk_paths`, fixed horizon t - t0. Elliptic PDEs (elliptic.py):
`fk_exit_paths`, run until the path leaves the domain.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

StepCallback = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None]


def brownian_increments(
    shape: tuple[int, ...],
    dt: float | torch.Tensor,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
    device=None,
    dtype=None,
    path_dim: int = 1,
) -> torch.Tensor:
    """sqrt(dt) * Z. `dt` scalar or broadcastable to `shape`.

    antithetic: draw half along `path_dim` (the M axis) and concatenate with the negation.
    """
    n = shape[path_dim]
    if antithetic:
        if n % 2:
            raise ValueError(f"antithetic requires an even path count, got {n}")
        half = list(shape)
        half[path_dim] = n // 2
        h = torch.randn(*half, generator=generator, device=device, dtype=dtype)
        z = torch.cat([h, -h], dim=path_dim)
    else:
        z = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return torch.as_tensor(dt, device=device, dtype=dtype).sqrt() * z


def _time_arg(v, n: int, dtype, device, name: str) -> tuple[torch.Tensor, bool]:
    """Float or (N,)/(N,1) tensor -> (N, 1). Second output: was it scalar?"""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.reshape(1, 1).to(dtype=dtype, device=device).expand(n, 1), True
        t = v.reshape(-1, 1).to(dtype=dtype, device=device)
        if t.shape[0] != n:
            raise ValueError(f"{name} has {t.shape[0]} entries, expected {n} (one per start point)")
        return t, False
    return torch.full((n, 1), float(v), dtype=dtype, device=device), True


def euler_maruyama(
    x0: torch.Tensor,
    drift: Callable,
    diffusion: Callable,
    t0: float | torch.Tensor,
    t1: float | torch.Tensor,
    n_steps: int,
    n_paths: int = 1,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
    return_paths: bool = True,
    callback: StepCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """X_{k+1} = X_k + b dt + sigma sqrt(dt) Z, uniform dt_i = (t1_i - t0_i) / n_steps.

    t0/t1: float, or per-point tensor (N,)/(N,1) (FK gives every start point its own horizon).
    Returns (paths, times): paths (N, M, K+1, d), or (N, M, 1, d) endpoint only when
    return_paths=False. times (K+1,) for scalar t0/t1, else (N, K+1).
    callback(k, x, t, dt) fires before each update with the state at step k -- use it to
    accumulate running functionals without storing paths. dt = 0 is allowed (path stays put).
    """
    N, d = x0.shape
    M, K = n_paths, n_steps
    if K < 1:
        raise ValueError(f"n_steps must be >= 1, got {K}")
    if antithetic and M % 2:
        raise ValueError("antithetic requires an even number of paths")

    dtype, device = x0.dtype, x0.device
    a, a_scalar = _time_arg(t0, N, dtype, device, "t0")
    b, b_scalar = _time_arg(t1, N, dtype, device, "t1")
    per_point = not (a_scalar and b_scalar)

    dt_vec = (b - a) / K  # (N, 1)
    if (dt_vec < 0).any():
        raise ValueError("t1 must be >= t0 elementwise")
    dt = dt_vec.view(N, 1, 1)

    steps = torch.arange(K + 1, dtype=dtype, device=device)
    times_full = a + steps * dt_vec  # (N, K+1)
    times = times_full if per_point else times_full[0]

    x = x0.unsqueeze(1).expand(N, M, d).clone()
    out = torch.empty(N, M, K + 1, d, dtype=dtype, device=device) if return_paths else None
    if out is not None:
        out[:, :, 0] = x

    for k in range(K):
        t = times_full[:, k].view(N, 1, 1)
        if callback is not None:
            callback(k, x, t, dt)
        dW = brownian_increments(
            (N, M, d), dt, generator=generator, antithetic=antithetic, device=device, dtype=dtype
        )
        x = x + drift(x, t) * dt + diffusion(x, t) * dW
        if out is not None:
            out[:, :, k + 1] = x

    return (out if out is not None else x.unsqueeze(2)), times


def fk_paths(
    pde,
    x: torch.Tensor,
    t: torch.Tensor,
    n_paths: int = 256,
    n_steps: int = 50,
    t_end: float | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
    return_paths: bool = True,
    callback: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feynman-Kac paths of `pde` started at (x, t), run backward in PDE time to t_end.

    x (N, d), t (N, 1). t_end: float or (N, 1), default pde.t0. The integrator clock r runs
    forward over [0, s], s = t - t_end; the coefficients are evaluated at PDE time t - r.

    callback(X, s_pde, dt), if given, fires before each step with the flattened state
    X (N*M, d), PDE time s_pde (N*M, 1) and dt (N, 1, 1).

    Returns (paths (N, M, K+1, d), times): times is the elapsed clock r, (N, K+1) per point.
    """
    N, d = x.shape
    M = n_paths
    if t_end is None:
        t_end = torch.full_like(t, pde.t0)
    t_end = torch.as_tensor(t_end, dtype=t.dtype, device=t.device).expand_as(t)
    s = (t - t_end).clamp_min(0.0)  # (N, 1) per-point horizon

    # PDE time t - r, flattened to (N*M, 1). Cached per step: drift, diffusion and the
    # callback all receive the same r object, so the copy is paid once per step.
    cache: dict = {"r": None, "tp": None}

    def pde_time(r):
        if cache["r"] is not r:
            cache["r"] = r
            cache["tp"] = (t.reshape(N, 1, 1) - r).expand(N, M, 1).reshape(N * M, 1)
        return cache["tp"]

    def b(X, r):
        return pde.drift(X.reshape(N * M, d), pde_time(r)).reshape(N, M, d)

    def sigma(X, r):
        return pde.diffusion(X.reshape(N * M, d), pde_time(r)).reshape(N, M, d)

    step_cb = None
    if callback is not None:

        def step_cb(k, X, r, dt):
            callback(X.reshape(N * M, d), pde_time(r), dt)

    return euler_maruyama(
        x0=x,
        drift=b,
        diffusion=sigma,
        t0=torch.zeros_like(s),
        t1=s,
        n_steps=n_steps,
        n_paths=M,
        generator=generator,
        antithetic=antithetic,
        return_paths=return_paths,
        callback=step_cb,
    )


def fk_exit_paths(
    pde,
    x: torch.Tensor,
    n_paths: int = 500,
    dt: float = 1e-3,
    t_max: float = math.inf,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Euler-Maruyama until exit, for an elliptic pde (paper Algorithm 1).

    Each of the M paths per start point x (N, d) steps X += b dt + sigma sqrt(dt) Z, is folded
    back across reflecting walls, and stops at the first step outside `pde.domain`, or once
    n dt >= t_max. Running terms accumulate left-Riemann, before each step:

        run  += disc * f(X) dt,      disc *= exp(-V(X) dt)

    Returns a dict: x_end (N, M, d) first point outside (last point if truncated), run and
    disc (N, M), tau (N, M) stopping time, exited (N, M) bool. Exits are only checked at grid
    times, which misses short excursions: bias O(sqrt(dt)). Start points outside the domain
    stop immediately with tau = 0.
    """
    N, d = x.shape
    M = n_paths
    NM = N * M
    dtype, device = x.dtype, x.device
    sqdt = math.sqrt(dt)

    X = x.repeat_interleave(M, dim=0)
    run = torch.zeros(NM, 1, dtype=dtype, device=device)
    disc = torch.ones(NM, 1, dtype=dtype, device=device)
    idx = torch.arange(NM, device=device)  # global index of each live path

    out = {
        "x_end": torch.empty(NM, d, dtype=dtype, device=device),
        "run": torch.empty(NM, 1, dtype=dtype, device=device),
        "disc": torch.empty(NM, 1, dtype=dtype, device=device),
        "tau": torch.empty(NM, dtype=dtype, device=device),
        "exited": torch.zeros(NM, dtype=torch.bool, device=device),
    }

    def retire(mask, n, exited):
        j = idx[mask]
        out["x_end"][j], out["run"][j], out["disc"][j] = X[mask], run[mask], disc[mask]
        out["tau"][j] = n * dt
        out["exited"][j] = exited

    n = 0
    gone = ~pde.domain.contains(X)
    while True:
        if gone.any():  # retire finished paths, keep the live ones compact
            retire(gone, n, True)
            keep = ~gone
            X, run, disc, idx = X[keep], run[keep], disc[keep], idx[keep]
        if idx.numel() == 0 or n * dt >= t_max:
            break
        if pde.inhomogeneous:
            run = run + disc * pde.source(X) * dt
            disc = disc * torch.exp(-pde.potential(X) * dt)
        Z = torch.randn(X.shape, generator=generator, dtype=dtype, device=device)
        X = pde.domain.reflect(X + pde.drift(X) * dt + pde.diffusion(X) * sqdt * Z)
        n += 1
        gone = ~pde.domain.contains(X)

    if idx.numel():  # truncated at t_max
        retire(torch.ones_like(idx, dtype=torch.bool), n, False)

    return {
        "x_end": out["x_end"].reshape(N, M, d),
        "run": out["run"].reshape(N, M),
        "disc": out["disc"].reshape(N, M),
        "tau": out["tau"].reshape(N, M),
        "exited": out["exited"].reshape(N, M),
    }
