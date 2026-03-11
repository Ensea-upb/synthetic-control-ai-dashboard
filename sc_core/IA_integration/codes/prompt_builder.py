def build_synth_analysis_prompt(
    variable: str,
    treated_unit: str,
    treatment_year: int
) -> str:

    return f"""
Tu es un économètre spécialiste de la méthode du contrôle synthétique.

Analyse le graphique fourni.

Variable analysée : {variable}
Unité traitée : {treated_unit}
Année du traitement : {treatment_year}

1. Décris l'ajustement pré-traitement.
2. Analyse la divergence post-traitement.
3. Mentionne les limites visibles.
4. Conclus prudemment.

Réponse en français, concise et factuelle.
"""