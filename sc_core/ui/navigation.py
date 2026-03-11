

from __future__ import annotations
import streamlit as st


PAGE_MAP = {
    "Accueil": "pages/0_Accueil.py",
    "Données": "pages/1_Donnees.py",
    "Exploration": "pages/2_Exploration.py",
    "Estimation": "pages/3_Estimation.py",
    "Résultat": "pages/4_Resultat.py",
    "Robustesse": "pages/5_Robustesse.py",
}

ORDERED_PAGES = [
    ("Accueil", "pages/0_Accueil.py"),
    ("Données", "pages/1_Donnees.py"),
    ("Exploration", "pages/2_Exploration.py"),
    ("Estimation", "pages/3_Estimation.py"),
    ("Résultat", "pages/4_Resultat.py"),
    ("Robustesse", "pages/5_Robustesse.py"),
]


def go_to(page_name: str) -> None:
    if page_name not in PAGE_MAP:
        raise ValueError(f"Unknown page_name='{page_name}'")
    st.switch_page(PAGE_MAP[page_name])


def render_top_navigation(current_page: str) -> None:
    cols = st.columns(len(ORDERED_PAGES))
    for col, (name, _) in zip(cols, ORDERED_PAGES):
        with col:
            disabled = (name == current_page)
            if st.button(name, use_container_width=True, disabled=disabled, key=f"nav_top_{name}"):
                go_to(name)


def render_prev_next(current_page: str) -> None:
    names = [name for name, _ in ORDERED_PAGES]
    idx = names.index(current_page)

    prev_name = names[idx - 1] if idx > 0 else None
    next_name = names[idx + 1] if idx < len(names) - 1 else None

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        if prev_name is not None:
            if st.button(f"⬅ {prev_name}", use_container_width=True, key=f"prev_{current_page}"):
                go_to(prev_name)

    with c3:
        if next_name is not None:
            if st.button(f"{next_name} ➜", use_container_width=True, key=f"next_{current_page}"):
                go_to(next_name)