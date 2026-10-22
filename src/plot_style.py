"""Shared matplotlib style and figure I/O. Use in every notebook so figures stay consistent.

    import plot_style
    plot_style.use()
    fig, ax = plot_style.subplots()
    ...
    plot_style.savefig(fig, "heat1d_convergence")   # figures/heat1d_convergence.{pdf,png}
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Repo-root figures/ folder, independent of the notebook's working directory.
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"

# Column widths for a two-column journal page, in inches.
SINGLE_COL = 3.4
DOUBLE_COL = 7.0

# Fixed method colours: the same method has the same colour in every figure.
COLORS: dict[str, str] = {
    "reference": "#333333",
    "analytic": "#333333",
    "mc_fk": "#0072B2",
    "pinn": "#D55E00",
    "fkpinn": "#009E73",
    "fkpinn_sig": "#CC79A7",
    "deep_bsde": "#8C564B",
}

# Colour-blind-safe cycle (Okabe-Ito) for everything not in COLORS.
CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

_RC: dict[str, Any] = {
    "figure.figsize": (SINGLE_COL, SINGLE_COL * 0.75),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
    # Vector text in the PDF, so figures stay searchable/selectable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use(latex: bool = False) -> None:
    """Apply the project style. `latex=True` needs a working TeX install."""
    mpl.rcParams.update(_RC)
    if latex:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )


def subplots(nrows: int = 1, ncols: int = 1, width: float = SINGLE_COL, aspect: float = 0.75, **kw):
    """plt.subplots sized in journal column units, constrained layout."""
    figsize = kw.pop("figsize", (width, width * aspect * nrows / max(ncols, 1)))
    return plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **kw)


def color(method: str) -> str:
    """Colour for a method name. Unknown names get a stable (not per-session) cycle colour."""
    return COLORS.get(method, CYCLE[zlib.crc32(method.encode()) % len(CYCLE)])


def savefig(fig, name: str, out_dir: str | Path = FIGURE_DIR, close: bool = True) -> list[Path]:
    """Save `<name>.pdf` (for LaTeX) and `<name>.png` (for slides)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def band(ax, x, ys, label: str | None = None, c: str | None = None, quantiles=(0.25, 0.75)):
    """Median line with an inter-quantile band over seeds. ys: (n_seeds, n_points)."""
    ys = np.asarray(ys)
    lo, hi = np.quantile(ys, quantiles[0], axis=0), np.quantile(ys, quantiles[1], axis=0)
    (line,) = ax.plot(x, np.median(ys, axis=0), label=label, color=c)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18, linewidth=0)
    return line


def reference_slope(
    ax, x0: float, y0: float, slope: float, span: float = 3.0, label: str | None = None
):
    """Dashed reference line of given slope on a log-log axis (e.g. MC error ~ M^-1/2)."""
    x1 = x0 * span
    y1 = y0 * (span**slope)
    ax.plot([x0, x1], [y0, y1], "k--", lw=0.9)
    if label:
        ax.annotate(
            label, ((x0 * x1) ** 0.5, (y0 * y1) ** 0.5), fontsize=7, ha="center", va="bottom"
        )
