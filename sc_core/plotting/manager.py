# centralise toutes les fonctions de plotting du projet

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


from .exploratory import (
    plot_dynamic_timeseries_wide,
    plot_static_bar_by_city,
    plot_missingness_heatmap,
    plot_obs_count_by_unit,
    plot_obs_count_by_time,
    plot_variable_histogram,
    plot_variable_boxplot,
)

from .diagnostique import (
    plot_control_envelope,
    plot_pre_treatment_distances,
)

from .plot_fit import (
    plot_donor_weights,
    plot_covariate_weights,
    plot_objective_history,
    plot_fit_summary_2x2,
)

from .effects import (
    plot_treated_vs_synthetic,
    plot_gap,
    plot_cumulative_gap,
)

from .robustness import (
    plot_placebo_gaps,
    plot_rmspe_distribution,
    plot_rmspe_ratio_ranking,
    plot_leave_one_out_gaps,
    plot_backdating_gaps,
    plot_backdating_ratio_bars,
)

from .exploratory import (
    plot_exploration_dynamic,
    plot_exploration_static,
)



class PlotManager:
    """
    Centralise toutes les fonctions de plotting du projet.
    """

    # =========================================================
    # EXPLORATORY
    # =========================================================

    @staticmethod
    def exploration_dynamic(
        df_wide,
        variable_name: str,
        treated_unit: str,
        intervention_time=None,
        control_units=None,
        show_envelope: bool = True,
    ):
        return plot_exploration_dynamic(
            df_wide=df_wide,
            variable_name=variable_name,
            treated_unit=treated_unit,
            intervention_time=intervention_time,
            control_units=control_units,
            show_envelope=show_envelope,
        )

    @staticmethod
    def exploration_static(**kwargs):
        return plot_exploration_static(**kwargs)

    @staticmethod
    def missingness(df: pd.DataFrame):
        return plot_missingness_heatmap(df)

    @staticmethod
    def obs_by_unit(df: pd.DataFrame, unit_col="ville"):
        return plot_obs_count_by_unit(df, unit_col)

    @staticmethod
    def obs_by_time(df: pd.DataFrame, time_col="date"):
        return plot_obs_count_by_time(df, time_col)

    @staticmethod
    def histogram(df: pd.DataFrame, col: str):
        return plot_variable_histogram(df, col)

    @staticmethod
    def boxplot(df: pd.DataFrame, col: str, unit_col="ville"):
        return plot_variable_boxplot(df, col, unit_col)

    # =========================================================
    # DIAGNOSTICS
    # =========================================================

    @staticmethod
    def control_envelope(df_wide: pd.DataFrame, treated: str):
        return plot_control_envelope(df_wide, treated)

    @staticmethod
    def pre_treatment_distances(df_wide: pd.DataFrame, treated: str, T0):
        return plot_pre_treatment_distances(df_wide, treated, T0)

    # =========================================================
    # FIT / ESTIMATION
    # =========================================================

    @staticmethod
    def donor_weights(weights: pd.Series):
        return plot_donor_weights(weights)

    @staticmethod
    def covariate_weights(weights: pd.Series):
        return plot_covariate_weights(weights)

    @staticmethod
    def objective_history(history):
        return plot_objective_history(history)

    @staticmethod
    def fit_summary(
        time_index,
        y_treated,
        y_synth,
        unit_weights,
        covariate_weights,
        objective_history,
        T0=None,
    ):
        return plot_fit_summary_2x2(
            time_index,
            y_treated,
            y_synth,
            unit_weights,
            covariate_weights,
            objective_history,
            T0,
        )

    # =========================================================
    # EFFECTS
    # =========================================================

    @staticmethod
    def treated_vs_synthetic(time_index, y_treated, y_synth, T0=None):
        return plot_treated_vs_synthetic(time_index, y_treated, y_synth, T0)

    @staticmethod
    def gap(time_index, y_treated, y_synth, T0=None):
        return plot_gap(time_index, y_treated, y_synth, T0)

    @staticmethod
    def cumulative_gap(time_index, y_treated, y_synth, T0=None):
        return plot_cumulative_gap(time_index, y_treated, y_synth, T0)

    # =========================================================
    # ROBUSTNESS
    # =========================================================

    @staticmethod
    def placebo_gaps(
        time_index,
        gaps_dict: Dict[str, np.ndarray],
        treated_unit: str,
        base_gap: Optional[np.ndarray] = None,
        T0=None,
    ):
        return plot_placebo_gaps(
            time_index=time_index,
            gaps_dict=gaps_dict,
            treated_unit=treated_unit,
            base_gap=base_gap,
            T0=T0,
        )

    @staticmethod
    def rmspe_distribution(rmspe_info):
        return plot_rmspe_distribution(rmspe_info)

    @staticmethod
    def rmspe_ratio(ratio_series: pd.Series, treated: Optional[str] = None):
        return plot_rmspe_ratio_ranking(ratio_series, treated)

    @staticmethod
    def leave_one_out_gaps(
        time_index,
        base_gap: np.ndarray,
        gaps_by_donor: Dict[str, np.ndarray],
        treated_label: str = "Baseline (tous donneurs)",
        T0=None,
    ):
        return plot_leave_one_out_gaps(
            time_index=time_index,
            base_gap=base_gap,
            gaps_by_donor=gaps_by_donor,
            treated_label=treated_label,
            T0=T0,
        )

    @staticmethod
    def backdating_gaps(time_index, runs: List[Any], real_T0=None):
        return plot_backdating_gaps(time_index=time_index, runs=runs, real_T0=real_T0)

    @staticmethod
    def backdating_ratio_bars(runs: List[Any]):
        return plot_backdating_ratio_bars(runs)
