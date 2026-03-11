

from __future__ import annotations
import streamlit as st


def require_data_loaded() -> bool:
    if st.session_state.get("df_raw", None) is None:
        st.warning("Aucune donnée chargée. Passe d'abord par la page Données.")
        return False
    return True


def require_sc_format() -> bool:
    if st.session_state.get("sc_format", None) is None:
        st.warning("Aucun format SCM disponible. Passe d'abord par la page Données.")
        return False
    return True


def require_estimation_result() -> bool:
    if st.session_state.get("estimation_result", None) is None:
        st.warning("Aucun résultat d'estimation disponible.")
        return False
    return True