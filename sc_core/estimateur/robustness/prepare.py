from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = value
    elif isinstance(value, tuple):
        out = list(value)
    else:
        try:
            out = list(value)
        except Exception:
            return []
    return [str(x) for x in out]


def get_pre_post_periods_from_scformat(sc_format) -> tuple[list, list, list]:
    years = list(np.asarray(sc_format.years).tolist())
    pre_mask = np.asarray(sc_format.pre_mask, dtype=bool)
    post_mask = np.asarray(sc_format.post_mask, dtype=bool)

    if len(years) != len(pre_mask) or len(years) != len(post_mask):
        raise ValueError(
            "SCFormat incohérent : years, pre_mask et post_mask doivent avoir la même longueur."
        )

    pre_periods = [years[i] for i in range(len(years)) if pre_mask[i]]
    post_periods = [years[i] for i in range(len(years)) if post_mask[i]]
    all_periods = pre_periods + post_periods
    return pre_periods, post_periods, all_periods


def build_robustness_panel_from_scformat(
    sc_format,
    *,
    include_outcome: bool = True,
    include_covariates: bool = True,
    drop_outcome_from_covariates: bool = True,
) -> pd.DataFrame:
    """
    Construit le panel backend canonique attendu par les wrappers robustesse.

    Format de sortie :
      - colonne 'vars'
      - colonne 'annee'
      - colonne 'ville_traite'
      - une colonne par donneur

    Le bloc outcome est reconstruit depuis Y1/Y0/years.
    Le bloc covariables est reconstruit depuis X_long.
    """
    treated = str(sc_format.treated)
    donors = _as_str_list(sc_format.donors)
    years = list(np.asarray(sc_format.years).tolist())

    if not treated:
        raise ValueError("SCFormat.treated est requis.")
    if not donors:
        raise ValueError("SCFormat.donors est requis.")
    if not years:
        raise ValueError("SCFormat.years est requis.")

    frames: list[pd.DataFrame] = []

    # --------------------------------------------------
    # Outcome block
    # --------------------------------------------------
    if include_outcome:
        y1 = np.asarray(sc_format.Y1, dtype=float).reshape(-1)
        y0 = np.asarray(sc_format.Y0, dtype=float)

        if y0.ndim != 2:
            raise ValueError("SCFormat.Y0 doit être une matrice 2D.")
        if len(years) != y0.shape[0] or len(y1) != y0.shape[0]:
            raise ValueError("Dimensions incohérentes entre years, Y1 et Y0.")
        if y0.shape[1] != len(donors):
            raise ValueError(
                "Nombre de donneurs incohérent entre SCFormat.donors et SCFormat.Y0."
            )

        outcome_df = pd.DataFrame(
            np.column_stack([y1, y0]),
            columns=["ville_traite"] + donors,
        )
        outcome_df.insert(0, "annee", years)
        outcome_df.insert(0, "vars", "y")
        frames.append(outcome_df)

    # --------------------------------------------------
    # Covariate block from X_long
    # --------------------------------------------------
    if include_covariates:
        x_long = getattr(sc_format, "X_long", None)
        if x_long is not None:
            cov_df = pd.DataFrame(x_long).copy()

            rename_map = {}
            if "var_name" in cov_df.columns:
                rename_map["var_name"] = "vars"
            if "date" in cov_df.columns:
                rename_map["date"] = "annee"
            if treated in cov_df.columns:
                rename_map[treated] = "ville_traite"

            cov_df = cov_df.rename(columns=rename_map)

            required_cols = ["vars", "annee", "ville_traite"] + donors
            missing = [c for c in required_cols if c not in cov_df.columns]
            if missing:
                raise ValueError(
                    "X_long ne permet pas de construire le panel robustesse. "
                    f"Colonnes manquantes : {missing}"
                )

            cov_df = cov_df[required_cols].copy()
            cov_df["vars"] = cov_df["vars"].astype(str)

            if drop_outcome_from_covariates:
                cov_df = cov_df.loc[cov_df["vars"] != "y"].copy()

            frames.append(cov_df)

    if not frames:
        raise ValueError("Aucun bloc à construire pour le panel robustesse.")

    panel_df = pd.concat(frames, axis=0, ignore_index=True)

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    duplicated = panel_df.duplicated(subset=["vars", "annee"], keep=False)
    if duplicated.any():
        dup_df = panel_df.loc[duplicated, ["vars", "annee"]].drop_duplicates()
        preview = dup_df.head(20).to_string(index=False)
        raise ValueError(
            "Duplicated (vars, annee) pairs detected in robustness panel:\n"
            f"{preview}"
        )

    panel_df = panel_df.sort_values(["vars", "annee"]).reset_index(drop=True)
    return panel_df


def select_valid_predictor_vars(
    panel_df: pd.DataFrame,
    *,
    predictor_vars: Sequence[str],
    pre_periods: Sequence,
) -> list[str]:
    """
    Retourne les prédicteurs ayant au moins une observation dans le pré-traitement
    (ou une valeur statique, annee=NaN).

    On n'exige PAS une couverture complète de toutes les pré-périodes :
    une covariable observée tous les 5 ou 10 ans est tout aussi valide
    pour le matching SCM — ses lignes disponibles seront simplement moins nombreuses.
    """
    predictor_vars = [str(x) for x in predictor_vars]
    pre_periods_set = set(pre_periods)

    valid: list[str] = []

    for pred in predictor_vars:
        block = panel_df.loc[panel_df["vars"].astype(str) == pred, ["annee"]].copy()
        if block.empty:
            continue

        # Covariable statique (annee=NaN) : toujours valide
        if block["annee"].isna().any():
            valid.append(pred)
            continue

        # Covariable dynamique : valide dès qu'une observation tombe dans le pré-traitement
        if len(pre_periods_set) == 0 or block["annee"].isin(pre_periods_set).any():
            valid.append(pred)

    return valid


def get_backend_outcome_and_predictors(sc_format, panel_df: pd.DataFrame):
    """
    Retourne (outcome_var, predictor_vars).
    predictor_vars est None si aucun prédicteur n'est disponible,
    ce qui déclenche le mode outcome-only dans build_prepared_matrices.
    """
    outcome_var = "y"

    raw_predictors = list(getattr(sc_format, "covariate_cols", []) or [])
    raw_predictors = [str(x) for x in raw_predictors]

    pre_periods, _, _ = get_pre_post_periods_from_scformat(sc_format)

    predictor_vars = select_valid_predictor_vars(
        panel_df,
        predictor_vars=raw_predictors,
        pre_periods=pre_periods,
    )

    # Fallback : si covariate_cols vide ou sans match, chercher dans le panel lui-même
    if not predictor_vars:
        all_panel_vars = [
            str(v) for v in panel_df["vars"].astype(str).unique()
            if str(v) != outcome_var
        ]
        predictor_vars = select_valid_predictor_vars(
            panel_df,
            predictor_vars=all_panel_vars,
            pre_periods=pre_periods,
        )

    # Retourner None pour signaler le mode outcome-only
    return outcome_var, predictor_vars if predictor_vars else None