"""Geometries for stationary (elliptic) problems: membership, sampling, reflection.

Interface
---------
    d, bounds (d, 2)          bounding box
    contains(x) -> (N,) bool  inside the domain (paths stop the first time this is False)
    reflect(x)  -> x          fold paths back across reflecting walls (identity if none)
    sample_interior(n)        uniform on the domain (rejection from the bounding box)
    sample_boundary(n)        uniform on the Dirichlet boundary
    grid(n)                   (xs, ys, points, inside) on an n x n grid, for plots / FD
"""

from __future__ import annotations

import math

import torch


def _box_faces(lo, hi, n, generator=None, device=None):
    """Uniform points on the faces of a box. Returns (x, axis, side)."""
    d = lo.shape[0]
    widths = hi - lo
    area = torch.stack([widths[torch.arange(d, device=lo.device) != i].prod() for i in range(d)])
    x = torch.rand(n, d, generator=generator, device=device) * widths + lo
    axis = torch.multinomial(area, n, replacement=True, generator=generator)
    side = torch.randint(0, 2, (n,), generator=generator, device=device)
    x[torch.arange(n, device=device), axis] = torch.where(side.bool(), hi[axis], lo[axis])
    return x, axis, side


class Domain:
    reflecting = False

    def __init__(self, bounds) -> None:
        self.bounds = torch.as_tensor(bounds, dtype=torch.float64)  # (d, 2)

    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    def _lohi(self, device=None, dtype=None):
        b = self.bounds.to(device=device, dtype=dtype or torch.get_default_dtype())
        return b[:, 0], b[:, 1]

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reflect(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def sample_boundary(self, n: int, generator=None, device=None) -> torch.Tensor:
        raise NotImplementedError

    def sample_interior(self, n: int, generator=None, device=None) -> torch.Tensor:
        lo, hi = self._lohi(device)
        out, have = [], 0
        while have < n:
            x = torch.rand(2 * (n - have) + 16, self.d, generator=generator, device=device)
            x = x * (hi - lo) + lo
            x = x[self.contains(x)]
            out.append(x)
            have += x.shape[0]
        return torch.cat(out)[:n]

    def grid(self, n: int = 201, dtype=None):
        """2-D grid over the bounding box. points (n*n, 2) in 'ij' order, inside (n, n)."""
        lo, hi = self._lohi(dtype=dtype)
        xs = torch.linspace(lo[0], hi[0], n, dtype=lo.dtype)
        ys = torch.linspace(lo[1], hi[1], n, dtype=lo.dtype)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        P = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        return xs, ys, P, self.contains(P).reshape(n, n)


class Box(Domain):
    """Axis-aligned box, all faces absorbing (Dirichlet). bounds [[lo_1, hi_1], ...]."""

    def contains(self, x):
        b = self.bounds.to(x)
        return ((x > b[:, 0]) & (x < b[:, 1])).all(dim=-1)

    def sample_boundary(self, n, generator=None, device=None):
        lo, hi = self._lohi(device)
        return _box_faces(lo, hi, n, generator, device)[0]


class Hexagon(Domain):
    """Regular hexagon centred at the origin, circumradius R, vertices at angles k*pi/3
    (pointy left/right, flat top/bottom at y = +-sqrt(3) R / 2)."""

    def __init__(self, R: float = 2.0) -> None:
        self.R = float(R)
        h = math.sqrt(3) / 2 * self.R
        super().__init__([[-self.R, self.R], [-h, h]])

    def vertices(self, device=None, dtype=None) -> torch.Tensor:
        a = torch.arange(6, dtype=torch.float64) * math.pi / 3
        v = self.R * torch.stack([a.cos(), a.sin()], dim=1)
        return v.to(device=device, dtype=dtype or torch.get_default_dtype())

    def contains(self, x):
        X, Y = x[:, 0].abs(), x[:, 1].abs()
        s3 = math.sqrt(3)
        return (Y < s3 / 2 * self.R) & (s3 * X + Y < s3 * self.R)

    def sample_boundary(self, n, generator=None, device=None):
        v = self.vertices(device)
        k = torch.randint(0, 6, (n,), generator=generator, device=device)
        s = torch.rand(n, 1, generator=generator, device=device)
        return v[k] + s * (v[(k + 1) % 6] - v[k])


class BoxMinusDisks(Domain):
    """Box with REFLECTING walls minus absorbing disks (2-D).

    The Dirichlet boundary is the union of the circles. The walls carry a zero-flux condition:
    paths are folded back into the box, and PINNs enforce it with `sample_walls` + a Neumann loss.
    """

    reflecting = True

    def __init__(self, bounds, centers, radii) -> None:
        super().__init__(bounds)
        self.centers = torch.as_tensor(centers, dtype=torch.float64)  # (K, 2)
        self.radii = torch.as_tensor(radii, dtype=torch.float64)  # (K,)

    def _gap(self, x):
        """Distance to each circle, negative inside a disk. (N, K)."""
        c, r = self.centers.to(x), self.radii.to(x)
        return (x[:, None, :] - c[None]).norm(dim=-1) - r

    def nearest_disk(self, x: torch.Tensor) -> torch.Tensor:
        return self._gap(x).argmin(dim=-1)

    def contains(self, x):
        b = self.bounds.to(x)
        in_box = ((x >= b[:, 0]) & (x <= b[:, 1])).all(dim=-1)
        return in_box & (self._gap(x) > 0).all(dim=-1)

    def reflect(self, x):
        """Fold once across each wall, then clamp: a huge Euler step (steep potential near a
        wall) can overshoot by more than the box width, and paths must never leave the box."""
        b = self.bounds.to(x)
        lo, hi = b[:, 0], b[:, 1]
        x = torch.where(x < lo, 2 * lo - x, x)
        x = torch.where(x > hi, 2 * hi - x, x)
        return torch.maximum(torch.minimum(x, hi), lo)

    def sample_boundary(self, n, generator=None, device=None):
        dtype = torch.get_default_dtype()
        c = self.centers.to(device=device, dtype=dtype)
        r = self.radii.to(device=device, dtype=dtype)
        k = torch.multinomial(r, n, replacement=True, generator=generator)  # prop. to circumference
        th = 2 * math.pi * torch.rand(n, generator=generator, device=device)
        return c[k] + r[k, None] * torch.stack([th.cos(), th.sin()], dim=1)

    def sample_walls(self, n, generator=None, device=None):
        """Uniform points on the box walls and their outward unit normals: (x, normals)."""
        lo, hi = self._lohi(device)
        x, axis, side = _box_faces(lo, hi, n, generator, device)
        normals = torch.zeros_like(x)
        normals[torch.arange(n, device=device), axis] = 2.0 * side.to(x.dtype) - 1.0
        return x, normals
