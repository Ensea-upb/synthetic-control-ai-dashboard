from __future__ import annotations

from typing import Any, List, Sequence
import numpy as np
import pandas as pd

from ..core.types import PreparedMatrices
from ..core.exceptions import DataPreparationError


REQUIRED_COLUMNS = {"vars", "annee", "ville_traite"}


def _get_donor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in REQUIRED_COLUMNS]


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataPreparationError(f"Missing required columns: {sorted(missing)}")

    donor_cols = _get_donor_columns(df)
    if len(donor_cols) == 0:
        raise DataPreparationError("No donor columns found.")

    if df.empty:
        raise DataPreparationError("Input dataframe is empty.")

    if df.duplicated(subset=["vars", "annee"]).any():
        dup = df.loc[df.duplicated(subset=["vars", "annee"], keep=False), ["vars", "annee"]]
        raise DataPreparationError(
            "Duplicated (vars, annee) pairs detected:\n"
            f"{dup.drop_duplicates().to_string(index=False)}"
        )


def build_prepared_matrices(
    df: pd.DataFrame,
    *,
    outcome_var: str,
    predictor_vars: Sequence[str] | None,
    pre_periods: Sequence[Any],
) -> PreparedMatrices:
    """
    Expected input structure:
    | vars | annee | ville_traite | donor1 | donor2 | ... |
    """
    _validate_input(df)

    df = df.copy()
    donor_names = _get_donor_columns(df)
    outcome_var = str(outcome_var)
    pre_periods = list(pre_periods)

    if len(pre_periods) == 0:
        raise DataPreparationError("pre_periods cannot be empty.")

    df_pre = df.loc[df["annee"].isin(pre_periods)].copy()
    if df_pre.empty:
        raise DataPreparationError("No rows found for the requested pre_periods.")

    if predictor_vars is None:
        predictor_vars = [
            str(v) for v in df_pre["vars"].astype(str).unique().tolist()
            if str(v) != outcome_var
        ]
    else:
        predictor_vars = [str(v) for v in predictor_vars]

    # Fallback outcome-only : si aucun prédicteur, on utilise l'outcome
    # lui-même comme feature (SCM classique : matching sur la trajectoire pré-traitement)
    if len(predictor_vars) == 0:
        predictor_vars = [outcome_var]

    # Outcome block
    df_y = df_pre.loc[df_pre["vars"].astype(str) == outcome_var].copy()
    if df_y.empty:
        raise DataPreparationError(f"No rows found for outcome_var='{outcome_var}'.")

    df_y = df_y.sort_values("annee")
    # On conserve uniquement les pré-périodes effectivement disponibles dans l'outcome
    Y1_pre = df_y["ville_traite"].to_numpy(dtype=float)
    Y0_pre = df_y[donor_names].to_numpy(dtype=float)

    # Predictor block
    # Les prédicteurs peuvent avoir une couverture temporelle partielle
    # (covariables observées tous les 5 ou 10 ans, ou uniquement sur certaines années).
    # On accepte TOUTES les lignes disponibles dans le pré-traitement sans exiger
    # une couverture complète : chaque ligne (var, annee) devient une feature de matching.
    rows_x1: List[float] = []
    rows_x0: List[np.ndarray] = []
    row_var: List[str] = []
    row_time: List[Any] = []

    for var in predictor_vars:
        df_var = df_pre.loc[df_pre["vars"].astype(str) == var].copy()
        if df_var.empty:
            # Si la covariable n'a aucune observation dans le pré-traitement,
            # on la saute silencieusement (elle sera exclue du matching)
            continue

        df_var = df_var.sort_values("annee")

        for _, row in df_var.iterrows():
            rows_x1.append(float(row["ville_traite"]))
            rows_x0.append(row[donor_names].to_numpy(dtype=float))
            row_var.append(var)
            row_time.append(row["annee"])

    X1 = np.asarray(rows_x1, dtype=float)
    X0 = np.vstack(rows_x0).astype(float)

    group_names = list(dict.fromkeys(row_var))

    return PreparedMatrices(
        X1=X1,
        X0=X0,
        Y1_pre=Y1_pre,
        Y0_pre=Y0_pre,
        row_var=row_var,
        row_time=row_time,
        group_names=group_names,
        donor_names=donor_names,
        pre_periods=sorted(pre_periods),
    )