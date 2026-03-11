from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .base import create_figure_ax, finalize_ax


def plot_control_envelope(df_wide: pd.DataFrame, treated: str):

    controls = [c for c in df_wide.columns if c != treated]

    low = df_wide[controls].min(axis=1)
    high = df_wide[controls].max(axis=1)

    fig, ax = create_figure_ax((12,7))

    ax.fill_between(df_wide.index, low, high, alpha=0.2, label="control envelope")

    if treated in df_wide:
        ax.plot(df_wide.index, df_wide[treated], color="red", label=treated)

    finalize_ax(ax, "Control envelope", "Time", "Value")
    ax.legend()
    return fig


def plot_pre_treatment_distances(df_wide: pd.DataFrame, treated: str, T0):

    pre = df_wide[df_wide.index < T0]

    distances = {}

    for c in df_wide.columns:
        if c == treated:
            continue

        diff = pre[treated] - pre[c]
        distances[c] = np.sqrt(np.mean(diff ** 2))

    s = pd.Series(distances).sort_values()

    fig, ax = create_figure_ax()
    s.plot(kind="bar", ax=ax)

    finalize_ax(ax, "Pre-treatment RMSPE", "Unit", "Distance")
    return fig