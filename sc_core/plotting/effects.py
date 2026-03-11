"""
sc_core.plotting.effects
========================
Publication-quality plots for the treatment effect analysis:
  - treated vs synthetic trajectory
  - instantaneous gap
  - cumulative gap
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .base import (
    create_figure_ax,
    finalize_ax,
    add_treatment_line,
    finalize_legend,
)


# ------------------------------------------------------------------ #
#  Colour palette (consistent across the application)
# ------------------------------------------------------------------ #
_C_TREATED   = "#e63946"   # red
_C_SYNTHETIC = "#1d3557"   # dark blue
_C_ZERO      = "#6c757d"   # grey
_C_FILL_POS  = "#a8dadc"   # light teal  (positive gap)
_C_FILL_NEG  = "#f4a261"   # orange      (negative gap)


def _as_1d(arr) -> np.ndarray:
    return np.asarray(arr, dtype=float).reshape(-1)


# ------------------------------------------------------------------ #
#  Treated vs Synthetic
# ------------------------------------------------------------------ #

def plot_treated_vs_synthetic(
    time_index,
    y_treated,
    y_synth,
    T0=None,
    treated_label: str = "Unité traitée",
    synth_label: str = "Contrôle synthétique",
) -> plt.Figure:
    """Overlay of treated and synthetic trajectories with T0 marker."""
    t  = _as_1d(time_index)
    yt = _as_1d(y_treated)
    ys = _as_1d(y_synth)

    fig, ax = create_figure_ax((12, 6))

    ax.plot(t, ys, color=_C_SYNTHETIC, linewidth=2.2,
            linestyle="--", label=synth_label, zorder=2)
    ax.plot(t, yt, color=_C_TREATED, linewidth=2.5,
            label=treated_label, zorder=3)

    if T0 is not None:
        add_treatment_line(ax, T0, label=f"Traitement (T0 = {T0})")
        pre_mask  = t < T0
        post_mask = t >= T0
        if pre_mask.any():
            ax.axvspan(float(t[pre_mask][0]),  float(t[pre_mask][-1]),
                       alpha=0.04, color="grey")
        if post_mask.any():
            ax.axvspan(float(t[post_mask][0]), float(t[post_mask][-1]),
                       alpha=0.06, color=_C_TREATED)

    finalize_ax(
        ax,
        title="Trajectoires : Traité vs Contrôle Synthétique",
        xlabel="Période",
        ylabel="Outcome",
    )
    finalize_legend(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Instantaneous gap
# ------------------------------------------------------------------ #

def plot_gap(
    time_index,
    y_treated,
    y_synth,
    T0=None,
    label: str = "Gap instantané",
) -> plt.Figure:
    """Instantaneous gap = Y_treated − Y_synthetic with fill above/below zero."""
    t   = _as_1d(time_index)
    gap = _as_1d(y_treated) - _as_1d(y_synth)

    fig, ax = create_figure_ax((12, 5))

    ax.fill_between(t, gap, 0,
                    where=(gap >= 0), color=_C_FILL_POS, alpha=0.45, label="Gap > 0")
    ax.fill_between(t, gap, 0,
                    where=(gap < 0),  color=_C_FILL_NEG, alpha=0.45, label="Gap < 0")

    ax.plot(t, gap, color=_C_TREATED, linewidth=2.2, label=label, zorder=3)
    ax.axhline(0, color=_C_ZERO, linewidth=1.2, linestyle="-")

    if T0 is not None:
        add_treatment_line(ax, T0, label=f"T0 = {T0}")

    finalize_ax(
        ax,
        title="Gap instantané (Traité − Synthétique)",
        xlabel="Période",
        ylabel="Gap",
    )
    finalize_legend(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Cumulative gap
# ------------------------------------------------------------------ #

def plot_cumulative_gap(
    time_index,
    y_treated,
    y_synth,
    T0=None,
    label: str = "Effet cumulé",
) -> plt.Figure:
    """Cumulative sum of the gap."""
    t       = _as_1d(time_index)
    gap     = _as_1d(y_treated) - _as_1d(y_synth)
    cum_gap = np.cumsum(gap)

    fig, ax = create_figure_ax((12, 5))

    ax.fill_between(t, cum_gap, 0,
                    where=(cum_gap >= 0), color=_C_FILL_POS, alpha=0.45,
                    label="Effet positif cumulé")
    ax.fill_between(t, cum_gap, 0,
                    where=(cum_gap < 0),  color=_C_FILL_NEG, alpha=0.45,
                    label="Effet négatif cumulé")

    ax.plot(t, cum_gap, color=_C_SYNTHETIC, linewidth=2.2, label=label, zorder=3)
    ax.axhline(0, color=_C_ZERO, linewidth=1.2, linestyle="-")

    if T0 is not None:
        add_treatment_line(ax, T0, label=f"T0 = {T0}")

    finalize_ax(
        ax,
        title="Effet cumulé (somme des gaps)",
        xlabel="Période",
        ylabel="Effet cumulé",
    )
    finalize_legend(ax)
    fig.tight_layout()
    return fig
