from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Validation utilities
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    df_name: str = "DataFrame",
) -> None:
    """
    Vérifie que les colonnes requises existent.

    Parameters
    ----------
    df : pd.DataFrame
    required_cols : Iterable[str]
    df_name : str
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def validate_non_empty(
    df: pd.DataFrame | pd.Series,
    name: str = "object",
) -> None:
    """
    Vérifie qu'un objet pandas n'est pas vide.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"{name} is empty.")


def validate_unit_present(
    series_or_df,
    unit: str,
    name: str = "object",
) -> None:
    """
    Vérifie qu'une unité est présente dans l'index ou les colonnes.
    """
    if isinstance(series_or_df, pd.Series):
        if unit not in series_or_df.index:
            raise ValueError(f"{unit} not found in {name}.")
    else:
        if unit not in series_or_df.columns:
            raise ValueError(f"{unit} not found in {name}.")


# ============================================================
# Numeric conversion
# ============================================================

def coerce_numeric(
    s: pd.Series,
) -> pd.Series:
    """
    Convertit une série en numérique de manière tolérante.

    - remplace virgules décimales
    - supprime espaces
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"\s+", "", regex=True),
        errors="coerce",
    )


# ============================================================
# Series preparation
# ============================================================

def prepare_series(
    x,
    name: Optional[str] = None,
) -> pd.Series:
    """
    Convertit entrée en pandas Series.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)

    if name is not None:
        s.name = name

    return s


def prepare_weights_series(
    weights,
    name: str = "weight",
) -> pd.Series:
    """
    Prépare une série de poids.

    - convertit en Series
    - convertit en float
    - supprime NaN
    """
    s = prepare_series(weights, name=name)
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()

    if s.empty:
        raise ValueError("Weights series is empty.")

    return s


# ============================================================
# Sorting utilities
# ============================================================

def sort_series_desc(
    s: pd.Series,
) -> pd.Series:
    """
    Trie une série par ordre décroissant.
    """
    return s.sort_values(ascending=False)


def drop_zero_weights(
    s: pd.Series,
    tol: float = 1e-10,
) -> pd.Series:
    """
    Supprime les poids quasi nuls.
    """
    return s.loc[s.abs() > tol]


# ============================================================
# Figure utilities
# ============================================================

def create_figure_ax(
    figsize: tuple[int, int] = (10, 6),
):
    """
    Crée figure et axe matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize_ax(
    ax,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = True,
):
    """
    Applique configuration standard à un axe.
    """
    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, alpha=0.3)


def add_treatment_line(
    ax,
    T0,
    label: Optional[str] = None,
):
    """
    Ajoute ligne verticale de traitement.
    """
    if T0 is None:
        return

    ax.axvline(
        x=T0,
        linestyle="--",
        linewidth=2,
        color="blue",
        alpha=0.7,
        label=label if label else f"Treatment ({T0})",
    )


# ============================================================
# Legend utilities
# ============================================================

def finalize_legend(
    ax,
    outside: bool = False,
):
    """
    Place la légende.
    """
    if outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
    else:
        ax.legend(frameon=True)