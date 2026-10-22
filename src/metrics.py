"""Error metrics (pred vs ref -> float) and Monte-Carlo estimator diagnostics.

Paper (2606.00643) naming: L2 abs = rmse, L2 rel = rel_l2, H1 abs/rel = h1_abs / h1_rel.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).double()


def rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """||pred - ref|| / ||ref||."""
    p, r = _flat(pred), _flat(ref)
    return float(torch.linalg.norm(p - r) / (torch.linalg.norm(r) + _EPS))


def rel_linf(pred: torch.Tensor, ref: torch.Tensor) -> float:
    p, r = _flat(pred), _flat(ref)
    return float((p - r).abs().max() / (r.abs().max() + _EPS))


def rmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    p, r = _flat(pred), _flat(ref)
    return float(torch.sqrt(torch.mean((p - r) ** 2)))


def h1_abs(pred, ref, dpred, dref) -> float:
    """sqrt(mean(|u - u_ref|^2 + ||grad u - grad u_ref||^2))  (paper eq. E.10). d*: (N, d)."""
    e = (_flat(pred) - _flat(ref)) ** 2 + ((dpred.detach() - dref.detach()).double() ** 2).sum(1)
    return float(e.mean().sqrt())


def h1_rel(pred, ref, dpred, dref) -> float:
    """h1_abs / sqrt(mean(|u_ref|^2 + ||grad u_ref||^2))  (paper eq. E.11)."""
    norm = (_flat(ref) ** 2 + (dref.detach().double() ** 2).sum(1)).mean().sqrt()
    return h1_abs(pred, ref, dpred, dref) / float(norm + _EPS)


def mc_summary(samples: torch.Tensor, dim: int = -1) -> dict[str, float]:
    """Mean, per-point variance and standard error of an MC estimator, averaged over points.

    samples: per-path realisations, e.g. Y (N, M) from fk_target, paths along `dim`.
    """
    s = samples.detach().double()
    n = s.shape[dim]
    var = s.var(dim=dim, unbiased=True)
    return {
        "mc_mean": float(s.mean()),
        "mc_var": float(var.mean()),
        "mc_se": float((var / n).sqrt().mean()),
        "mc_n": float(n),
    }
