# IA_integration/context_builder.py

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from app_ui.state import keys


def _safe_repr(obj: Any, max_len: int = 1200) -> str:
    try:
        text = repr(obj)
    except Exception:
        text = f"<unrepresentable {type(obj).__name__}>"
    if len(text) > max_len:
        text = text[:max_len] + "... [truncated]"
    return text


def build_page_context(page_name: str) -> Dict[str, Any]:
    """
    Build a compact context payload for the current page.

    The objective is to avoid passing raw session_state blindly.
    """
    context = {
        "page_name": page_name,
        "data_config": st.session_state.get(keys.DATA_CONFIG),
        "estimation_config": st.session_state.get(keys.ESTIMATION_CONFIG),
        "workflow": st.session_state.get(keys.WORKFLOW),
    }

    if page_name in {"Donnees", "Exploration", "Estimation", "Resultat", "Robustesse"}:
        sc_format = st.session_state.get(keys.SC_FORMAT)
        context["sc_format_summary"] = _safe_repr(sc_format)

    if page_name in {"Resultat", "Robustesse"}:
        estimation_result = st.session_state.get(keys.ESTIMATION_RESULT)
        fit_summary = st.session_state.get(keys.FIT_SUMMARY_DATA)
        context["estimation_result_summary"] = _safe_repr(estimation_result)
        context["fit_summary_data_summary"] = _safe_repr(fit_summary)

        # Inline RMSPE digest for richer AI context
        try:
            import numpy as np
            t_idx = np.asarray(fit_summary.time_index, dtype=float)
            y_t   = np.asarray(fit_summary.y_treated,  dtype=float).reshape(-1)
            y_s   = np.asarray(fit_summary.y_synth,    dtype=float).reshape(-1)
            T0    = fit_summary.T0
            if T0 is not None:
                pre_mask  = t_idx <  T0
                post_mask = t_idx >= T0
                g = y_t - y_s
                pre_rmspe  = float(np.sqrt(np.mean(g[pre_mask]  ** 2))) if pre_mask.any()  else None
                post_rmspe = float(np.sqrt(np.mean(g[post_mask] ** 2))) if post_mask.any() else None
                ratio = (post_rmspe / pre_rmspe) if (pre_rmspe and pre_rmspe > 0) else None
                context["rmspe_digest"] = {
                    "T0":         T0,
                    "pre_rmspe":  round(pre_rmspe,  6) if pre_rmspe  is not None else None,
                    "post_rmspe": round(post_rmspe, 6) if post_rmspe is not None else None,
                    "ratio":      round(ratio, 4)      if ratio       is not None else None,
                }
        except Exception:
            pass

    if page_name == "Robustesse":
        robustness_results = st.session_state.get(keys.ROBUSTNESS_RESULTS)
        context["robustness_results_summary"] = _safe_repr(robustness_results)

    return context