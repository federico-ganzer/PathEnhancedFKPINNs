"""Truncated path signatures in pure PyTorch (batched, differentiable).

For a piecewise-linear path the signature of one segment with increment a is the tensor
exponential (level k = a^{(x)k} / k!), and Chen's relation S(X * Y) = S(X) (x) S(Y) glues
segments. Levels are flattened; the leading 1 is dropped, so the output length is
sum_{k=1..depth} d^k.

These are Stratonovich (piecewise-linear) signatures. Ito coordinates need a conversion.
"""

from __future__ import annotations

import torch


def signature_length(d: int, depth: int) -> int:
    """sum_{k=1..depth} d^k."""
    if d < 1 or depth < 1:
        raise ValueError("d and depth must be >= 1")
    total, term = 0, 1
    for _ in range(depth):
        term *= d
        total += term
    return total


def tensor_exp(increment: torch.Tensor, depth: int) -> list[torch.Tensor]:
    """Signature of one linear segment: level k is increment^{(x)k} / k!. Levels (..., d**k)."""
    levels = [increment]
    for k in range(2, depth + 1):
        prev = levels[-1]
        outer = prev.unsqueeze(-1) * increment.unsqueeze(-2)  # (..., d**(k-1), d)
        levels.append(outer.reshape(*prev.shape[:-1], -1) / k)
    return levels


def tensor_product(a: list[torch.Tensor], b: list[torch.Tensor], depth: int) -> list[torch.Tensor]:
    """Truncated tensor-algebra product (Chen). Level n = a_n + b_n + sum_{i+j=n} a_i (x) b_j."""
    out = []
    for n in range(1, depth + 1):
        term = a[n - 1] + b[n - 1]
        for i in range(1, n):
            outer = a[i - 1].unsqueeze(-1) * b[n - i - 1].unsqueeze(-2)
            term = term + outer.reshape(*a[i - 1].shape[:-1], -1)
        out.append(term)
    return out


def signature(path: torch.Tensor, depth: int, flatten: bool = True):
    """path (..., K+1, d) -> (..., signature_length(d, depth)), or a list of levels."""
    dX = path[..., 1:, :] - path[..., :-1, :]  # (..., K, d)
    acc = tensor_exp(dX[..., 0, :], depth)
    for k in range(1, dX.shape[-2]):
        acc = tensor_product(acc, tensor_exp(dX[..., k, :], depth), depth)
    return torch.cat(acc, dim=-1) if flatten else acc


def expected_signature(paths: torch.Tensor, depth: int, dim: int = 1) -> torch.Tensor:
    """Mean signature over the path axis `dim` of the signature output.

    (N, M, K+1, d) with dim=1 -> (N, L). Deterministic given the start point: a function of
    (x, t), not a per-path feature.
    """
    return signature(paths, depth).mean(dim=dim)


def time_augment(path: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
    """Prepend time as channel 0: (..., K+1, d) -> (..., K+1, d+1). Default times: [0, 1].

    `times` must broadcast to path.shape[:-1]; for fk_paths output (paths (N, M, K+1, d),
    times (N, K+1)) pass times[:, None]. Without time augmentation the signature is
    reparametrisation-invariant and loses all timing information.
    """
    if times is None:
        times = torch.linspace(0, 1, path.shape[-2], dtype=path.dtype, device=path.device)
    times = times.expand(*path.shape[:-1])
    return torch.cat([times.unsqueeze(-1), path], dim=-1)


def add_basepoint(path: torch.Tensor) -> torch.Tensor:
    """Prepend a zero point so absolute position is kept: (..., K+1, d) -> (..., K+2, d)."""
    return torch.cat([torch.zeros_like(path[..., :1, :]), path], dim=-2)
