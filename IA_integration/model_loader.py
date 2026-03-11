# IA_integration/model_loader.py
"""
Singleton de chargement du modèle local via st.cache_resource.

Le décorateur @st.cache_resource garantit qu'une seule instance
du AIManager est créée pour toute la durée de vie du serveur Streamlit,
quelle que soit la page visitée ou le nombre de reruns.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

# Chemin par défaut du modèle — relatif à la racine sc_app
_DEFAULT_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent / "local_models" / "Qwen2.5-VL-7B-Instruct-int4-ov"
)


@st.cache_resource(show_spinner=False)
def get_local_ai_manager(
    model_dir: Optional[str] = None,
    device: str = "CPU",
):
    """
    Charge et retourne le AIManager local (singleton Streamlit).

    Le modèle est chargé une seule fois au premier appel ;
    les appels suivants retournent l'instance en cache.

    Parameters
    ----------
    model_dir : chemin vers le dossier du modèle OpenVINO
    device    : "CPU", "GPU", "AUTO"
    """
    from sc_core.IA_integration.codes.manager import AIManager as CoreAIManager
    from sc_core.IA_integration.codes.status import AIStatus

    dir_to_use = model_dir or os.environ.get("SC_AI_MODEL_DIR", _DEFAULT_MODEL_DIR)

    manager = CoreAIManager(model_dir=dir_to_use, device=device)
    manager.load_blocking()   # chargement synchrone une seule fois
    return manager