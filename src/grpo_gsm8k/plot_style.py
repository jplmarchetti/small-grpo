"""House matplotlib style for report-ready plots (copied from the cowork-plot-style template).

Import this once at the top of any plotting script so every figure that lands
in a report shares the same look — same serif font (Palatino, so it matches the
Pages document), same palette, same annotation style, same export settings.

Usage:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import apply_house_style, C, br, label_points, label_bars, save_fig

    apply_house_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, y, "-o", color=C["green"], lw=2.6, ms=9, label="...")
    label_points(ax, x, y, color=C["green"])
    save_fig(fig, "meu_grafico.png")  # save wherever the context wants

The font falls back gracefully: real "Palatino" is used on the user's Mac; in a
Linux sandbox the metric-identical clone "TeX Gyre Pagella" (or P052) is picked
automatically, so the exported PNG looks the same either way.
"""
from __future__ import annotations

import logging

import matplotlib as mpl

# On machines without real Palatino, matplotlib logs a findfont warning for every
# text element as it falls through to the clone. The fallback is intended, so
# quiet that specific logger — keeps plot output clean without hiding real errors.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Palatino first (macOS), then metric-compatible clones for headless/Linux.
_FONT_STACK = [
    "Palatino",
    "Palatino Linotype",
    "TeX Gyre Pagella",
    "P052",
    "URW Palladio L",
    "serif",
]

# House palette. Keep meanings stable across every report so a color always
# means the same thing to the reader.
C = {
    "green": "#2e7d32",   # valor real / acerto / bom
    "red": "#c62828",     # previsão / erro / atenção
    "orange": "#e08214",  # previsão crua / destaque / categoria A
    "blue": "#1565c0",    # volume / disponibilidade / série secundária
    "purple": "#6a51a3",  # série terciária (ex.: "todos os dados")
    "grey": "#bdbdbd",    # resto / neutro em barras empilhadas
    "muted_red": "#d98a8a",  # fatia negativa suave (ex.: indisponível)
    "text": "#222222",
}


def apply_house_style() -> None:
    """Set global rcParams for the house look. Call once before plotting."""
    mpl.rcParams.update({
        "font.family": _FONT_STACK,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.95,
        "figure.titlesize": 14.5,
        "figure.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def br(n: float, decimals: int = 0) -> str:
    """Brazilian number formatting: thousands with '.', decimals with ','.

    br(1042223)      -> '1.042.223'
    br(5.13, 1)      -> '5,1'
    """
    s = f"{n:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def label_points(ax, xs, ys, color="#222222", above=True, fmt="{:.1f}%",
                 dy=10, fontsize=10):
    """Annotate each (x, y) point with its value, bold, above/below the marker.

    Direct labels beat a reader hunting the y-axis — every point in a report
    chart should be readable without measuring against the gridlines.
    """
    off = dy if above else -dy - 6
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=fontsize,
                    color=color, fontweight="bold")


def label_bars(ax, xs, values, bottoms=None, color="#222222", fmt="{:.1f}%",
               inside=True, fontsize=10):
    """Label bar segments. If bottoms given, centers label inside the segment
    (good for stacked bars); otherwise places it just above the bar top."""
    for i, x in enumerate(xs):
        v = values[i]
        if v is None:
            continue
        if bottoms is not None and inside:
            y = bottoms[i] + v / 2
            va = "center"
            ax.annotate(fmt.format(v), (x, y), ha="center", va=va,
                        fontsize=fontsize, color=color, fontweight="bold")
        else:
            base = 0 if bottoms is None else bottoms[i]
            ax.annotate(fmt.format(v), (x, base + v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=fontsize,
                        color=color, fontweight="bold")


def save_fig(fig, path: str, tight: bool = True) -> str:
    """Save with house export defaults and print the path. Returns the path."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass  # GridSpec figures with twin axes can warn; export still fine
    fig.savefig(path)
    print("saved", path)
    return path
