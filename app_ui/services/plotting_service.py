# app_ui/services/plotting_service.py
"""
Thin service layer between the UI and sc_core.plotting.

All figure-building functions return matplotlib Figure objects.
Use `fig_to_png_bytes(fig)` to get a bytes buffer suitable for
st.download_button.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd

from sc_core.plotting.manager import PlotManager


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_plot_manager() -> Any:
    try:
        return PlotManager()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate PlotManager: {type(exc).__name__}: {exc}"
        ) from exc


def _normalize_weights(weights) -> pd.Series:
    if weights is None:
        return pd.Series(dtype=float)
    if isinstance(weights, pd.Series):
        return weights.astype(float)
    if isinstance(weights, dict):
        return pd.Series(weights, dtype=float)
    try:
        return pd.Series(weights, dtype=float)
    except Exception as exc:
        raise ValueError(f"Cannot normalize weights to pandas Series: {exc}") from exc


def _extract_fit_summary_parts(fit_summary_data: Any) -> Dict[str, Any]:
    """Adapt FitSummaryData to the explicit argument contract of the plotting layer."""
    if fit_summary_data is None:
        raise ValueError("fit_summary_data is required.")
    return {
        "time_index":        getattr(fit_summary_data, "time_index", None),
        "y_treated":         getattr(fit_summary_data, "y_treated", None),
        "y_synth":           getattr(fit_summary_data, "y_synth", None),
        "unit_weights":      _normalize_weights(getattr(fit_summary_data, "unit_weights", None)),
        "covariate_weights": _normalize_weights(getattr(fit_summary_data, "covariate_weights", None)),
        "objective_history": getattr(fit_summary_data, "objective_history", None),
        "T0":                getattr(fit_summary_data, "T0", None),
    }


# ---------------------------------------------------------------------------
# Export utility
# ---------------------------------------------------------------------------

def fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib Figure to PNG bytes (for st.download_button)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Main estimation fit figures
# ---------------------------------------------------------------------------

def build_main_fit_figure(fit_summary_data: Any):
    pm = get_plot_manager()
    p  = _extract_fit_summary_parts(fit_summary_data)
    return pm.fit_summary(
        time_index=p["time_index"],
        y_treated=p["y_treated"],
        y_synth=p["y_synth"],
        unit_weights=p["unit_weights"],
        covariate_weights=p["covariate_weights"],
        objective_history=p["objective_history"],
        T0=p["T0"],
    )


def build_treated_vs_synthetic_plot(fit_summary_data: Any):
    pm = get_plot_manager()
    p  = _extract_fit_summary_parts(fit_summary_data)
    return pm.treated_vs_synthetic(
        time_index=p["time_index"],
        y_treated=p["y_treated"],
        y_synth=p["y_synth"],
        T0=p["T0"],
    )


def build_gap_plot(fit_summary_data: Any):
    pm = get_plot_manager()
    p  = _extract_fit_summary_parts(fit_summary_data)
    return pm.gap(
        time_index=p["time_index"],
        y_treated=p["y_treated"],
        y_synth=p["y_synth"],
        T0=p["T0"],
    )


def build_cumulative_gap_plot(fit_summary_data: Any):
    pm = get_plot_manager()
    p  = _extract_fit_summary_parts(fit_summary_data)
    return pm.cumulative_gap(
        time_index=p["time_index"],
        y_treated=p["y_treated"],
        y_synth=p["y_synth"],
        T0=p["T0"],
    )


def build_donor_weights_plot(weights: Any):
    pm = get_plot_manager()
    return pm.donor_weights(_normalize_weights(weights))


def build_covariate_weights_plot(weights: Any):
    pm = get_plot_manager()
    return pm.covariate_weights(_normalize_weights(weights))


def build_objective_history_plot(history: Any):
    pm = get_plot_manager()
    return pm.objective_history(history)


# ---------------------------------------------------------------------------
# Exploration figures
# ---------------------------------------------------------------------------

def build_exploration_dynamic_plot(**payload):
    pm = get_plot_manager()
    return pm.exploration_dynamic(
        df_wide=payload["df_wide"],
        variable_name=payload["variable_name"],
        treated_unit=payload["treated_unit"],
        control_units=payload["control_units"],
        intervention_time=payload["intervention_time"],
        show_envelope=payload.get("show_envelope", True),
    )


def build_exploration_static_plot(**payload):
    pm = get_plot_manager()
    return pm.exploration_static(
        df_static=payload["df_static"],
        variable_name=payload["variable_name"],
        treated_unit=payload["treated_unit"],
    )
