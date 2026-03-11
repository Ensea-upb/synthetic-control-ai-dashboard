# pages/1_Donnees.py

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_ui.components.ai_panel import render_ai_panel
from app_ui.controllers.data_controller import (
    build_data_config,
    build_sc_format_from_config,
    build_sc_format_summary,
    extract_available_units_and_years,
    infer_candidate_columns,
    load_multiple_uploaded_dataframes,
    merge_uploaded_dataframes,
    persist_built_sc_format,
    persist_loaded_dataframe,
    standardize_merge_keys,
    validate_data_config,
)
from app_ui.state import keys
from app_ui.state.initialization import initialize_app_state
from app_ui.state.workflow import recompute_workflow_from_state
from app_ui.state.invalidation import invalidate_from_data_change

# Optional: use native project sidebar if already available
try:
    from sc_core.ui.sidebar import render_workflow_sidebar
except Exception:
    render_workflow_sidebar = None


st.set_page_config(page_title="Données | Synthetic Control App", layout="wide")
initialize_app_state()
st.session_state[keys.CURRENT_PAGE] = "Donnees"
recompute_workflow_from_state()

if render_workflow_sidebar is not None:
    render_workflow_sidebar()

st.title("2) Données — Import et configuration SCM")
st.caption("Cette page construit le SCFormat, qui devient l’entrée unique du pipeline d’estimation.")

# =========================================================
# Upload block
# =========================================================
with st.expander("Importer un ou plusieurs fichiers", expanded=True):
    uploaded_files = st.file_uploader(
        "Choisir un ou plusieurs fichiers de données",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
    )

    merge_how = st.selectbox(
        "Type de fusion",
        options=["outer", "inner", "left", "right"],
        index=0,
        help="outer = conserve toutes les observations ; inner = intersection stricte.",
    )

    col_up_1, col_up_2, col_up_3 = st.columns(3)

    with col_up_1:
        load_clicked = st.button("Charger les bases", type="primary", width="stretch")

    with col_up_2:
        merge_clicked = st.button("Fusionner les bases", width="stretch")

    with col_up_3:
        clear_clicked = st.button("Effacer les données chargées", width="stretch")

    if clear_clicked:
        invalidate_from_data_change()
        st.session_state[keys.DF_RAW] = None
        st.session_state[keys.DFS_BY_NAME] = {}
        st.session_state[keys.PREPARED_DFS_BY_NAME] = {}
        st.session_state[keys.DATA_CONFIG] = None
        st.session_state[keys.DATA_SUMMARY] = None
        recompute_workflow_from_state()
        st.success("Les données chargées et tous les objets aval ont été effacés.")

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")

    # -------------------------
    # ACTION 1 : CHARGER
    # -------------------------
    if load_clicked:
        if not uploaded_files:
            st.error("Tu dois d’abord sélectionner au moins un fichier.")
        else:
            try:
                dfs_by_name = load_multiple_uploaded_dataframes(uploaded_files)
                st.session_state[keys.DFS_BY_NAME] = dfs_by_name
                st.success(
                    "Bases chargées avec succès. "
                    "Configure maintenant les colonnes ville/date pour chaque base."
                )
            except Exception as exc:
                st.error(f"Erreur lors du chargement : {exc}")

    dfs_by_name = st.session_state.get(keys.DFS_BY_NAME)

    # -------------------------
    # ETAPE 2 : MAPPING PAR FICHIER
    # -------------------------
    if dfs_by_name:
        st.subheader("Configuration des clés par base")

        prepared_dfs_by_name = {}

        for file_name, df_file in dfs_by_name.items():
            with st.expander(f"Base : {file_name}", expanded=False):
                st.write(f"Dimensions : {df_file.shape[0]} lignes × {df_file.shape[1]} colonnes")
                st.dataframe(df_file.head(10), width="stretch")

                cols = list(df_file.columns)

                default_city = "ville" if "ville" in cols else cols[0]
                default_date_candidates = [c for c in cols if c.lower() in {"date", "annee", "year"}]
                default_date = default_date_candidates[0] if default_date_candidates else cols[min(1, len(cols) - 1)]

                city_col_file = st.selectbox(
                    f"Colonne ville — {file_name}",
                    options=cols,
                    index=cols.index(default_city),
                    key=f"city_col_{file_name}",
                )

                date_col_file = st.selectbox(
                    f"Colonne date — {file_name}",
                    options=cols,
                    index=cols.index(default_date),
                    key=f"date_col_{file_name}",
                )

                try:
                    prepared_df = standardize_merge_keys(
                        df_file,
                        city_col=city_col_file,
                        date_col=date_col_file,
                    )
                    prepared_dfs_by_name[file_name] = prepared_df
                    st.caption("Colonnes standardisées pour la fusion : 'ville' et 'date'.")
                except Exception as exc:
                    st.error(f"Erreur de standardisation pour {file_name} : {exc}")

        st.session_state[keys.PREPARED_DFS_BY_NAME] = prepared_dfs_by_name

    prepared_dfs_by_name = st.session_state.get(keys.PREPARED_DFS_BY_NAME)

    # -------------------------
    # ACTION 2 : FUSIONNER
    # -------------------------
    if merge_clicked:
        if not prepared_dfs_by_name:
            st.error("Charge d’abord les bases et configure les colonnes ville/date.")
        else:
            try:
                df_raw = merge_uploaded_dataframes(
                    prepared_dfs_by_name,
                    how=merge_how,
                )
                persist_loaded_dataframe(df_raw)
                recompute_workflow_from_state()
                st.success("Bases fusionnées avec succès.")
            except Exception as exc:
                st.error(f"Erreur lors de la fusion : {exc}")
df_raw = st.session_state.get(keys.DF_RAW)
# =========================================================
# Data preview
# =========================================================
if df_raw is not None:
    st.subheader("Aperçu des données")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Nombre de lignes", int(df_raw.shape[0]))
    with c2:
        st.metric("Nombre de colonnes", int(df_raw.shape[1]))
    with c3:
        st.metric("Valeurs manquantes", int(df_raw.isna().sum().sum()))

    with st.expander("Voir un extrait des données", expanded=False):
        st.dataframe(df_raw.head(20), use_container_width=True)

    with st.expander("Voir les types de colonnes", expanded=False):
        dtype_df = pd.DataFrame(
            {
                "column": df_raw.columns,
                "dtype": [str(dt) for dt in df_raw.dtypes],
                "missing_count": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
            }
        )
        st.dataframe(dtype_df, use_container_width=True)

# =========================================================
# Configuration block
# =========================================================
if df_raw is not None:
    st.subheader("Configuration du problème SCM")

    inferred = infer_candidate_columns(df_raw)
    all_cols = inferred["all"]

    default_city_col = inferred["city_candidates"][0] if inferred["city_candidates"] else all_cols[0]
    default_date_col = inferred["date_candidates"][0] if inferred["date_candidates"] else all_cols[0]
    default_y_col = inferred["y_candidates"][0] if inferred["y_candidates"] else all_cols[0]

    cfg_prev = st.session_state.get(keys.DATA_CONFIG) or {}

    c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
    with c_cfg1:
        city_col = st.selectbox(
            "Colonne unité (ville / région / pays)",
            options=all_cols,
            index=all_cols.index(cfg_prev.get("city_col", default_city_col)),
        )
    with c_cfg2:
        date_col = st.selectbox(
            "Colonne temps / date",
            options=all_cols,
            index=all_cols.index(cfg_prev.get("date_col", default_date_col)),
        )
    with c_cfg3:
        y_col = st.selectbox(
            "Variable outcome Y",
            options=all_cols,
            index=all_cols.index(cfg_prev.get("y_col", default_y_col)),
        )

    try:
        available_units, available_years = extract_available_units_and_years(
            df=df_raw,
            city_col=city_col,
            date_col=date_col,
        )
    except Exception as exc:
        st.error(f"Impossible d’extraire les unités / années : {exc}")
        available_units, available_years = [], []

    if len(available_units) == 0 or len(available_years) == 0:
        st.warning("La configuration courante ne permet pas encore d’identifier correctement les unités et les années.")
    else:
        st.subheader("Définition du traitement")

        c_t1, c_t2 = st.columns(2)

        prev_treated = cfg_prev.get("treated_city", available_units[0])
        treated_idx = available_units.index(prev_treated) if prev_treated in available_units else 0

        with c_t1:
            treated_city = st.selectbox(
                "Unité traitée",
                options=available_units,
                index=treated_idx,
            )

        donor_default = cfg_prev.get(
            "donor_cities",
            [u for u in available_units if u != treated_city][: min(5, max(0, len(available_units) - 1))],
        )

        donor_default = [u for u in donor_default if u in available_units and u != treated_city]

        with c_t2:
            donor_cities = st.multiselect(
                "Pool de donneurs",
                options=[u for u in available_units if u != treated_city],
                default=donor_default,
            )

        prev_treatment_year = cfg_prev.get("treatment_year", available_years[len(available_years) // 2])
        if prev_treatment_year not in available_years:
            prev_treatment_year = available_years[len(available_years) // 2]

        treatment_year = st.selectbox(
            "Année de traitement T0",
            options=available_years,
            index=available_years.index(prev_treatment_year),
        )

        pre_years_default = [y for y in available_years if y < treatment_year]
        post_years_default = [y for y in available_years if y >= treatment_year]

        st.subheader("Features utilisées pour l’ajustement")

        c_f1, c_f2 = st.columns(2)

        with c_f1:
            y_feature_years = st.multiselect(
                "Années outcome utilisées dans X",
                options=available_years,
                default=cfg_prev.get("y_feature_years", pre_years_default),
                help="Ces années du outcome sont injectées dans les features d’ajustement.",
            )

        candidate_covs = [c for c in all_cols if c not in {city_col, date_col, y_col}]

        with c_f2:
            covariate_cols = st.multiselect(
                "Covariables candidates",
                options=candidate_covs,
                default=cfg_prev.get("covariate_cols", candidate_covs[: min(5, len(candidate_covs))]),
            )

        cov_feature_default = cfg_prev.get("cov_feature_years", pre_years_default)

        cov_feature_years = st.multiselect(
            "Années des covariables dynamiques utilisées dans X",
            options=available_years,
            default=cov_feature_default,
            help="Pour les covariables temporelles, seules ces années seront utilisées.",
        )

        st.subheader("Normalisation")

        c_n1, c_n2, c_n3 = st.columns(3)
        with c_n1:
            normalize_X = st.checkbox(
                "Normaliser X",
                value=cfg_prev.get("normalize_X", True),
            )
        with c_n2:
            normalize_method = st.selectbox(
                "Méthode de normalisation",
                options=["robust", "zscore"],
                index=0 if cfg_prev.get("normalize_method", "robust") == "robust" else 1,
                disabled=not normalize_X,
            )
        with c_n3:
            allow_empty_features = st.checkbox(
                "Autoriser X vide",
                value=cfg_prev.get("allow_empty_features", False),
                help="À activer uniquement pour des cas spéciaux de debug ou outcome-only.",
            )

        config = build_data_config(
            city_col=city_col,
            date_col=date_col,
            y_col=y_col,
            treated_city=treated_city,
            donor_cities=donor_cities,
            treatment_year=int(treatment_year),
            covariate_cols=covariate_cols,
            y_feature_years=y_feature_years,
            cov_feature_years=cov_feature_years,
            normalize_X=normalize_X,
            normalize_method=normalize_method,
            allow_empty_features=allow_empty_features,
        )

        validation = validate_data_config(df_raw, config)

        with st.expander("Résumé de la configuration courante", expanded=False):
            st.json(config)

        if not validation.ok:
            for err in validation.errors:
                st.error(err)

        build_clicked = st.button(
            "Construire le SCFormat",
            type="primary",
            use_container_width=True,
            disabled=not validation.ok,
        )

        if build_clicked:
            try:
                sc_format = build_sc_format_from_config(df_raw, config)
                persist_built_sc_format(config, sc_format)
                st.success("SCFormat construit avec succès.")
            except Exception as exc:
                st.error(f"Erreur pendant la construction du SCFormat : {exc}")

# =========================================================
# SCFormat summary
# =========================================================
sc_format = st.session_state.get(keys.SC_FORMAT)

if sc_format is not None:
    st.subheader("Résumé du SCFormat construit")

    summary = build_sc_format_summary(sc_format)

    c_s1, c_s2, c_s3, c_s4, c_s5 = st.columns(5)
    with c_s1:
        st.metric("Périodes", summary.get("n_periods"))
    with c_s2:
        st.metric("Donneurs", summary.get("n_donors"))
    with c_s3:
        st.metric("Features", summary.get("n_features"))
    with c_s4:
        st.metric("Pré-traitement", summary.get("n_pre_periods"))
    with c_s5:
        st.metric("Post-traitement", summary.get("n_post_periods"))

    with st.expander("Détails techniques", expanded=False):
        st.json(summary)

# =========================================================
# Page-level AI assistant
# =========================================================
st.divider()

render_ai_panel(
    page_name="Donnees",
    available_actions=[
        "Expliquer ma configuration",
        "Interpréter l'exploration",
    ],
)