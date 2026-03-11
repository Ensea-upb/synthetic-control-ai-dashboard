"""
sc_core
=======

Core package for the synthetic control application.

This package centralizes:
- data preparation for SCM
- estimation methods
- robustness analyses
- results post-processing
- plotting utilities
"""

from .data_management import SCFormat, build_sc_format

from .estimateur import (
    fit_xv_random_search_from_df,
    fit_xv_bilevel_from_df,
    fit_xv_trainval_from_df,
    fit_xv_random_search_from_scformat,
    fit_xv_bilevel_from_scformat,
    fit_xv_trainval_from_scformat,
    compute_rmspe_metrics,
    compute_rmspe_ratio_series,
    run_placebo_space,
    run_leave_one_out,
    run_backdating,
)

from .results import (
    FitSummaryData,
    build_synthetic_series,
    build_gap,
    build_cumulative_gap,
    build_unit_weights_series,
    build_covariate_weights_series,
    build_fit_summary_data,
    build_result_dict_for_ui,
)

__all__ = [
    "SCFormat",
    "build_sc_format",
    "fit_xv_random_search_from_df",
    "fit_xv_bilevel_from_df",
    "fit_xv_trainval_from_df",
    "fit_xv_random_search_from_scformat",
    "fit_xv_bilevel_from_scformat",
    "fit_xv_trainval_from_scformat",
    "compute_rmspe_metrics",
    "compute_rmspe_ratio_series",
    "run_placebo_space",
    "run_leave_one_out",
    "run_backdating",
    "FitSummaryData",
    "build_synthetic_series",
    "build_gap",
    "build_cumulative_gap",
    "build_unit_weights_series",
    "build_covariate_weights_series",
    "build_fit_summary_data",
    "build_result_dict_for_ui",
]