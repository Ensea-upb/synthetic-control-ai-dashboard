# plotting functions for synthetic control fit diagnostics

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

from .base import create_figure_ax, finalize_ax, sort_series_desc, drop_zero_weights


def plot_donor_weights(weights: pd.Series):

    w = sort_series_desc(weights)

    fig, ax = create_figure_ax()
    w.plot(kind="bar", ax=ax)

    finalize_ax(ax, "Donor weights", "Unit", "Weight")
    return fig


def plot_covariate_weights(weights: pd.Series):

    w = sort_series_desc(weights)

    fig, ax = create_figure_ax()
    w.plot(kind="bar", ax=ax)

    finalize_ax(ax, "Covariate weights", "Covariate", "Weight")
    return fig


def plot_objective_history(history):

    fig, ax = create_figure_ax()

    ax.plot(history, marker="o")

    finalize_ax(ax, "Objective convergence", "Iteration", "Loss")
    return fig


def plot_fit_summary_2x2(
    time_index,
    y_treated,
    y_synth,
    unit_weights,
    covariate_weights,
    objective_history,
    T0=None,
):

    fig, axes = plt.subplots(2,2, figsize=(16,10))

    ax = axes[0,0]
    ax.plot(time_index, y_treated, label="treated", color="red")
    ax.plot(time_index, y_synth, label="synthetic")

    if T0 is not None:
        ax.axvline(T0, linestyle="--")

    ax.legend()
    ax.set_title("Outcome fit")

    ax = axes[0,1]
    sort_series_desc(unit_weights).plot(kind="bar", ax=ax)
    ax.set_title("Donor weights")

    ax = axes[1,0]
    sort_series_desc(covariate_weights).plot(kind="bar", ax=ax)
    ax.set_title("Covariate weights")

    ax = axes[1,1]
    ax.plot(objective_history)
    ax.set_title("Objective history")

    plt.tight_layout()

    return fig