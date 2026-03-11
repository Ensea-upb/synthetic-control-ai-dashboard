from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

from .base import (
    validate_columns,
    validate_non_empty,
    coerce_numeric,
    finalize_ax,
    create_figure_ax,
)


def plot_dynamic_timeseries_wide(df_wide: pd.DataFrame, treated: str, T0=None):
    validate_non_empty(df_wide, "df_wide")

    fig, ax = create_figure_ax((12,7))

    controls = [c for c in df_wide.columns if c != treated]

    for c in controls:
        ax.plot(df_wide.index, df_wide[c], alpha=0.6)

    if treated in df_wide.columns:
        ax.plot(df_wide.index, df_wide[treated], color="red", linewidth=3, label=treated)

    if T0 is not None:
        ax.axvline(T0, linestyle="--", color="blue")

    finalize_ax(ax, "Time series", "Time", "Value")
    return fig


def plot_static_bar_by_city(df: pd.DataFrame, value_col: str, unit_col="ville", treated=None):
    validate_columns(df, [unit_col, value_col])

    d = df.copy()
    d[value_col] = coerce_numeric(d[value_col])
    d = d.sort_values(value_col, ascending=False)

    fig, ax = create_figure_ax((12,7))

    colors = ["red" if v == treated else "skyblue" for v in d[unit_col]]
    ax.bar(d[unit_col], d[value_col], color=colors)

    finalize_ax(ax, value_col, "Units", value_col)
    plt.xticks(rotation=45)
    return fig


def plot_missingness_heatmap(df: pd.DataFrame):
    validate_non_empty(df)

    fig, ax = create_figure_ax((12,6))
    plt.heatmap(df.isna(), cbar=False, ax=ax)

    finalize_ax(ax, "Missing values heatmap")
    return fig


def plot_obs_count_by_unit(df: pd.DataFrame, unit_col="ville"):
    validate_columns(df, [unit_col])

    counts = df.groupby(unit_col).size().sort_values()

    fig, ax = create_figure_ax()
    counts.plot(kind="bar", ax=ax)

    finalize_ax(ax, "Observations by unit", "Unit", "Count")
    return fig


def plot_obs_count_by_time(df: pd.DataFrame, time_col="date"):
    validate_columns(df, [time_col])

    counts = df.groupby(time_col).size().sort_index()

    fig, ax = create_figure_ax()
    counts.plot(ax=ax)

    finalize_ax(ax, "Observations by time", "Time", "Count")
    return fig


def plot_variable_histogram(df: pd.DataFrame, col: str):
    validate_columns(df, [col])

    fig, ax = create_figure_ax()
    plt.hist(coerce_numeric(df[col]), ax=ax, bins=30)

    finalize_ax(ax, f"Distribution {col}", col, "Frequency")
    return fig


def plot_variable_boxplot(df: pd.DataFrame, col: str, unit_col="ville"):

    validate_columns(df, [col, unit_col])

    fig, ax = create_figure_ax((12,6))
    plt.boxplot([df[df[unit_col] == u][col] for u in df[unit_col].unique()], labels=df[unit_col].unique(), ax=ax)

    plt.xticks(rotation=45)
    finalize_ax(ax, f"Boxplot {col}")
    return fig


def plot_exploration_dynamic(
    df_wide: pd.DataFrame,
    variable_name: str,
    treated_unit: str,
    control_units,
    intervention_time=None,
    show_envelope: bool = True
):

    fig, ax = plt.subplots(figsize=(12, 7))

    controls = [c for c in control_units if c in df_wide.columns]

    df_wide[controls].plot(ax=ax, linewidth=2, alpha=0.8)

    ax.plot(
        df_wide.index,
        df_wide[treated_unit],
        linewidth=3,
        color="red",
        label=treated_unit,
    )

    X = df_wide[controls].apply(pd.to_numeric, errors="coerce")

    low = X.min(axis=1)
    high = X.max(axis=1)

    if show_envelope and controls:
        ax.fill_between(
            df_wide.index,
            low,
            high,
            alpha=0.1,
            color="red",
            label="Enveloppe contrôles",
        )

    if intervention_time is not None:
        ax.axvline(
            x=intervention_time,
            linestyle="--",
            color="blue",
            linewidth=2,
            label="Traitement",
        )

    ax.set_title(f"Evolution de {variable_name}")
    ax.set_xlabel("Temps")
    ax.set_ylabel(variable_name)

    ax.grid(True, alpha=0.3)


    # légende à droite (extrême droite)
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

    return fig


def plot_exploration_static(
    df_static: pd.DataFrame,
    variable_name: str,
    treated_unit: str,
):

    df = df_static.sort_values("value", ascending=False)

    colors = [
        "red" if str(c) == str(treated_unit) else "skyblue"
        for c in df["ville"]
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(df["ville"], df["value"], color=colors)

    ax.set_title(f"Niveau de {variable_name}")
    ax.set_xlabel("Ville")
    ax.set_ylabel(variable_name)

    plt.xticks(rotation=45)

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    return fig