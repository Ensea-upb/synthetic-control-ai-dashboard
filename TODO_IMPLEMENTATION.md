# TODO_IMPLEMENTATION.md — SC-APP
> Synthetic Control Application · Suivi d'implémentation
> Dernière mise à jour : 2026-03-10

---

## Légende

| Icône | Signification |
|-------|---------------|
| ✅ | Terminé et validé |
| 🔧 | En cours / partiellement fait |
| ❌ | Non commencé |
| 🐛 | Bug connu |
| ⚠️ | Point de vigilance |
| 🔬 | Requis pour validation scientifique |
| 🤖 | Lié à l'intégration IA |

---

## Sommaire

1. [Socle de données — SCFormat](#1-socle-de-données--scformat)
2. [Optimisation — Solveurs internes](#2-optimisation--solveurs-internes)
3. [Optimisation — Solveurs externes](#3-optimisation--solveurs-externes)
4. [Tests de robustesse](#4-tests-de-robustesse)
5. [Interface utilisateur — Pages](#5-interface-utilisateur--pages)
6. [Gestion d'état et workflow](#6-gestion-détat-et-workflow)
7. [Visualisations](#7-visualisations)
8. [Intégration IA](#8-intégration-ia)
9. [Tests unitaires et validation scientifique](#9-tests-unitaires-et-validation-scientifique)
10. [Extensions scientifiques (Phase 2)](#10-extensions-scientifiques-phase-2)
11. [Productionisation (Phase 3)](#11-productionisation-phase-3)

---

## 1. Socle de données — SCFormat

**Fichier :** `sc_core/data_management/sc_format.py`

### 1.1 Dataclass et champs

- ✅ Champs de base : `city_col`, `date_col`, `y_col`, `treated`, `donors`, `years`, `T0`
- ✅ Matrices outcome : `Y1 (T,)`, `Y0 (T, J)`
- ✅ Matrices features : `X1 (K,)`, `X0 (K, J)`
- ✅ Masques temporels : `pre_mask`, `post_mask`
- ✅ Métadonnées features : `row_var`, `group_names`, `feature_names`
- ✅ Champ `feature_names: List[str]` ajouté (bug fix session 2)
- ✅ Normalisation : `normalize_X: bool`, `normalize_method: str`
- ✅ `X_long: pd.DataFrame` avec colonnes `var_name`, `date`, `unit`
- ✅ `covariate_cols`, `y_feature_years`, `cov_feature_years`

### 1.2 Invariants

- ✅ I1 : `len(years) == len(Y1) == nrows(Y0) == T`
- ✅ I2 : `len(donors) == ncols(Y0) == ncols(X0) == J`
- ✅ I3 : `X1.shape[0] == X0.shape[0] == K`
- ✅ I4/I5 : `pre_mask ∧ post_mask == 0`, `pre_mask ∨ post_mask == 1`
- ✅ I6 : `sum(pre_mask) == T0 >= 2`
- ❌ Lever des exceptions typées (`SCFormatError`) avec messages lisibles par l'UI

### 1.3 Pipeline de construction

**Fichier :** `sc_core/estimateur/utils/data_prep.py`

- ✅ Pivot DataFrame brut → matrices Y
- ✅ Construction des masques temporels
- ✅ Gestion covariables à couverture irrégulière (bug fix session 2)
- ✅ `select_valid_predictor_vars()` : accepte covariable avec ≥ 1 obs en pré-traitement
- ✅ `get_backend_outcome_and_predictors()` retourne `None` (pas `[]`) pour mode outcome-only
- ✅ Normalisation robuste (médiane / IQR, fallback IQR=0 → 1)
- ✅ Normalisation z-score
- 🔧 Rapport de validation retourné à l'UI (structure partielle, affichage incomplet)
- ❌ Détecter et signaler les unités avec trop de NaN en outcome
- ❌ Détecter et signaler les donneurs quasi-identiques (corrélation > 0.99)
- ⚠️ Vérifier comportement si `T0 == 1` (doit lever I6 avant toute estimation)

### 1.4 Normalisation

**Fichier :** `sc_core/estimateur/utils/scaling.py`

- ✅ Mode `robust`
- ✅ Mode `zscore`
- ✅ Mode `none`
- ❌ Sauvegarder les paramètres de normalisation pour dénormalisation des résultats
- ⚠️ Vérifier que la normalisation est appliquée **avant** construction de X1/X0 et non après

---

## 2. Optimisation — Solveurs internes

**Fichier :** `sc_core/estimateur/methods/inner/`

### 2.1 SLSQP (`slsqp.py`)

- ✅ Construction de H = X0ᵀ V X0, f = X0ᵀ V X1
- ✅ Appel `scipy.optimize.minimize(..., method='SLSQP')`
- ✅ Projection simplexe post-optimisation : `max(W,0) / ‖max(W,0)‖₁`
- ✅ Initialisation uniforme W⁰ = 1/J
- ❌ Retourner le flag de convergence dans le résultat
- ⚠️ Cas H singulière ou mal conditionnée : lever `SingularMatrixError` typée

### 2.2 QP direct KKT (`qp_direct.py`)

- ✅ Résolution système KKT sur ensemble actif
- ✅ Itération sur l'ensemble actif A = {j : wⱼ > 0}
- 🔧 Stabilité numérique sur panels avec donneurs très corrélés (cas non testé)
- ❌ Tests unitaires sur cas limites (J=1, J=2, tous poids nuls)

---

## 3. Optimisation — Solveurs externes

**Fichier :** `sc_core/estimateur/methods/outer/`

### 3.1 Random Search (`random_search.py`)

- ✅ Échantillonnage Dirichlet(1_K) sur Δ^(K-1)
- ✅ Graine reproductible via `np.random.default_rng(seed)`
- ✅ Callback temps réel : émettre `(n, ℓ*, W*, V*)` à chaque amélioration
- ✅ Historique des losses `{ℓₙ}`
- ❌ Paramètre `early_stopping` si pas d'amélioration sur N_patience itérations

### 3.2 Bilevel (`bilevel.py`)

- ✅ Multi-restarts R avec init Dirichlet
- ✅ Support méthodes Powell et L-BFGS-B
- ✅ Projection `max(v,0) / ‖max(v,0)‖₁` dans la fonction objectif
- 🔧 Callback partiel (pas émis à chaque restart)
- ❌ Exposer `maxiter` par restart dans la config UI
- ⚠️ Risque de surfit documenté — usage déconseillé sur petits panels

### 3.3 TrainVal (`trainval.py`)

**Fichier :** `sc_core/estimateur/validation/time_split.py` (découpage)

- ✅ Partition pré-traitement en Train / Validation
- ✅ Paramètre `val_last_k` (défaut k=3)
- ✅ Re-estimation sur l'intégralité du pré-traitement après sélection de V*
- ❌ Lever une exception claire si `T0 - k < 2` (trop peu de périodes d'entraînement)
- ❌ Log de la loss de validation par itération (distinct de la loss d'entraînement)

### 3.4 Orchestrateur (`estimator.py`)

- ✅ Dispatch vers la méthode sélectionnée
- ✅ Retour `EstimationResult` unifié
- 🔧 Gestion des timeouts (non implémentée)
- ❌ Sauvegarder l'`EstimationResult` complet incluant historique et config

---

## 4. Tests de robustesse

**Répertoire :** `sc_core/estimateur/robustness/`

> Tous les fichiers ont été réécrits / corrigés en session 2.

### 4.1 Placebo espace (`placebo_space.py`)

- ✅ Boucle sur chaque donneur comme unité traitée fictive
- ✅ Construction sous-panel `P_j` (traité fictif = j, donneurs = tous sauf j et 1)
- ✅ Calcul RMSPE_pré, RMSPE_post, ratio rⱼ pour chaque placebo
- ✅ p-valeur : `#{j : rⱼ ≥ r₁} / J`
- ✅ Retour `PlaceboResult` avec `gaps_dict`, `rmspe_info`, `ratio_series`, `treated_unit`
- ❌ Filtrage des placebos avec RMSPE_pré >> RMSPE_pré(traité) (optionnel, standard Abadie)
- ❌ Test de cohérence : vérifier que `r₁` est correctement calculé avec le résultat de référence

### 4.2 RMSPE (`rmspe.py`)

- ✅ RMSPE_pré, RMSPE_post, ratio r
- ✅ Ranking des ratios (traité inclus)
- ✅ Affichage p-valeur

### 4.3 Leave-One-Out (`leave_one_out.py`)

- ✅ Identification des donneurs actifs `A = {j : wⱼ* > 1e-10}`
- ✅ Re-estimation sans chaque donneur actif
- ✅ Retour `LOOResult` avec `gaps_by_donor`, `base_result`
- ❌ Gérer le cas où `|A| == 1` (un seul donneur actif, LOO = aucun donneur)
- ⚠️ Coût computationnel : `|A|` estimations complètes — prévoir indicateur de durée

### 4.4 Backdating (`backdating.py`)

- ✅ Génération de la liste des pseudo-T₀ intérieurs
- ✅ Skip si `|pré(T̃₀)| < 2`
- ✅ Re-estimation avec fenêtre pré réduite à `[1, T̃₀)`
- ✅ Calcul ratio backdating `r^(T̃₀)` = RMSPE([T̃₀, T₀)) / RMSPE([1, T̃₀))
- ✅ Retour liste de `BackdatingRun` : `pseudo_t0`, `gaps`, `pre_periods_used`
- ❌ Stopper si le nombre de pseudo-T₀ valides < 2 (résultat non interprétable)

### 4.5 Prepare (`prepare.py`)

- ✅ Construction des `SCFormat` dérivés pour chaque sous-test
- 🔧 Réutilisation de la configuration de normalisation du résultat de référence
- ❌ Passer explicitement la méthode et les hyperparamètres d'estimation de référence

---

## 5. Interface utilisateur — Pages

### 5.0 Page 0 — Accueil (`pages/0_Accueil.py`)

- ✅ 5 métriques de workflow (`st.metric`)
- ✅ Appel `trigger_ai_model_loading()` au chargement
- ✅ Statut modèle IA (idle / loading / ready / error)
- ✅ Bouton reset global
- 🔧 Description pédagogique du SCM (placeholder présent, texte incomplet)
- ❌ Lien vers documentation externe ou aide contextuelle
- ❌ Afficher la version de l'app et date de dernière mise à jour

### 5.1 Page 1 — Données (`pages/1_Donnees.py`)

- ✅ Upload multi-fichiers CSV/XLSX
- ✅ Détection automatique des colonnes candidates
- ✅ Sélection `city_col`, `date_col`, `y_col`
- ✅ Sélection unité traitée et donneurs
- ✅ Sélection T₀
- ✅ Sélection covariables et années de features
- ✅ Mode normalisation
- ✅ Construction du `SCFormat`
- 🔧 Rapport de validation affiché (incomplet sur certains edge cases)
- ❌ Gestion des doublons avec choix utilisateur (garder premier / dernier / moyenne)
- ❌ Prévisualisation des données fusionnées avant construction du SCFormat
- ❌ Avertissement si T₀ laisse moins de 3 périodes en pré ou post
- ❌ Avertissement si un donneur a plus de 20% de NaN sur l'outcome

### 5.2 Page 2 — Exploration (`pages/2_Exploration.py`)

- ✅ Ajout dynamique de graphiques par variable
- ✅ Enveloppe min-max des contrôles
- ✅ Ligne verticale T₀ configurable
- ✅ Commentaire IA par graphique (`exploration_chart_comment`)
- 🔧 Panel IA : réponse persistée dans `session_state[f"ai_resp_{page_name}"]`
- ❌ Option pour superposer plusieurs variables sur un même graphique
- ❌ Statistiques descriptives pré/post par unité (tableau)
- ❌ Test de tendances parallèles (visuel) avant T₀

### 5.3 Page 3 — Estimation (`pages/3_Estimation.py`)

- ✅ Sélection méthode : Random Search / Bilevel / TrainVal
- ✅ Paramètres par méthode : `n_iter`, `n_restarts`, `val_last_k`, `seed`
- ✅ Barre de progression et callback temps réel
- ✅ Affichage dynamique du fit en cours de convergence
- ✅ Résumé final : loss, poids non nuls, statut
- ❌ Bouton d'annulation de l'estimation en cours
- ❌ Comparaison de plusieurs runs côte à côte (optionnel)
- ❌ Avertissement si loss finale anormalement élevée (seuil configurable)
- ⚠️ L'invalidation doit se déclencher dès que l'utilisateur change un paramètre,
  pas seulement après relancement de l'estimation

### 5.4 Page 4 — Résultats (`pages/4_Resultat.py`)

- ✅ Trajectoires observée vs synthétique
- ✅ Gap instantané τ̂ₜ
- ✅ Effet cumulé τ̂ₜᶜᵘᵐ
- ✅ Poids unitaires (bar chart trié)
- ✅ Poids covariables (bar chart par groupe)
- 🔧 Résumé numérique du fit (RMSPE pré/post, ratio)
- ❌ Export des figures (PNG / SVG)
- ❌ Export du résultat complet (CSV : années, Y₁, Ŷ₁, gap, gap cumulé)
- ❌ Table des poids avec valeurs numériques
- ❌ Interprétation IA de la page résultats (`results` task)

### 5.5 Page 5 — Robustesse (`pages/5_Robustesse.py`)

> Réécrite en session 2 (fix column unpacking, predictor coverage, covariables irrégulières).

- ✅ Section Placebo espace : gaps gris vs traité rouge
- ✅ Section RMSPE : bar chart pré/post, ranking, p-valeur
- ✅ Section Leave-One-Out : gaps colorés par donneur vs baseline
- ✅ Section Backdating : gradient viridis par pseudo-T₀
- ✅ `st.columns(2)` (corrigé depuis `st.columns(3)` qui causait une erreur d'unpacking)
- 🔧 Tableaux récapitulatifs numériques (structure présente, formatage à affiner)
- ❌ Exécution indépendante de chaque test (boutons séparés)
- ❌ Bouton "Lancer tous les tests"
- ❌ Indicateur de progression par test
- ❌ Interprétation IA de la page robustesse (`robustness` task)
- ❌ Export des figures de robustesse

---

## 6. Gestion d'état et workflow

### 6.1 Clés d'état (`app_ui/state/keys.py`)

- ✅ `SC_FORMAT`, `ESTIMATION_CONFIG`, `ESTIMATION_RESULT`
- ✅ `FIT_SUMMARY_DATA`, `ROBUSTNESS_CONFIG`, `ROBUSTNESS_RESULTS`
- ✅ `WORKFLOW`, `AI_ENABLED`, `AI_BACKEND`, `AI_LAST_RESPONSE`, `AI_CONTEXT_CACHE`
- ❌ Aucune chaîne littérale de clé en dehors de `keys.py` (à auditer)

### 6.2 Initialisation (`app_ui/state/initialization.py`)

- ✅ Initialisation de toutes les clés au premier chargement
- ✅ `trigger_ai_model_loading()` : lance le préchargement du modèle en arrière-plan

### 6.3 Workflow (`app_ui/state/workflow.py`)

- ✅ Calcul des flags de progression (données prêtes / estimation faite / etc.)
- 🔧 Vérification exhaustive de tous les prérequis par page
- ❌ Flag `estimation_stale` : levé quand la config change après une estimation existante

### 6.4 Invalidation (`app_ui/state/invalidation.py`)

- ✅ Propagation en cascade : Données → Estimation → Résultats → Robustesse
- 🔧 L'invalidation n'est pas toujours déclenchée lors d'un changement de paramètre
  d'estimation (sans rechargement de page)
- ❌ Message explicite affiché à l'utilisateur lors d'une invalidation

---

## 7. Visualisations

**Répertoire :** `sc_core/plotting/`

### 7.1 Manager (`manager.py`)

- ✅ `PlotManager` centralisé
- 🔧 Cache des figures (éviter recalcul sur re-render Streamlit)
- ❌ Découplage complet calcul scientifique / rendu graphique

### 7.2 Figures résultats (`results.py`)

- ✅ Trajectoires, gap, effet cumulé, poids
- ❌ Intervalle de confiance visuel (Phase 2)
- ❌ Option `dark_mode` pour export publication

### 7.3 Figures robustesse (`robustness.py`)

- ✅ Placebo space, LOO, backdating
- 🔧 Légende dynamique selon le nombre d'unités
- ❌ Zoom interactif sur la fenêtre post-traitement

---

## 8. Intégration IA

### 8.1 CoreAIManager (`sc_core/IA_integration/codes/manager.py`)

- ✅ `generate_text(prompt)` — inférence texte seul
- ✅ `comment_figure(fig, prompt)` — inférence vision + texte
- ✅ `load_blocking()` : `VLMPipeline(model_dir, device)`
- ✅ `graceful ImportError` si `openvino_genai` absent

### 8.2 ModelLoader (`IA_integration/model_loader.py`)

- ✅ Singleton `@st.cache_resource` : chargé une seule fois par session
- ✅ Chemin par défaut : `sc_app/local_models/Qwen2.5-VL-7B-Instruct-int4-ov/`
- ✅ Variable d'environnement `SC_AI_MODEL_DIR` pour surcharge
- ✅ Paramètre `device` (défaut `"CPU"`)
- ❌ Timeout de chargement avec message d'erreur clair si > 120s
- ❌ Test de santé post-chargement (génération d'un token test)

### 8.3 AIManager facade (`IA_integration/ai_manager.py`)

- ✅ Dispatch vers `comment_figure()` ou `generate_text()` selon présence de figure
- ✅ Retour `AIResponse(ok, content, error)`
- ❌ Retry automatique (1 fois) en cas d'erreur d'inférence

### 8.4 PromptBuilder (`IA_integration/prompt_builder.py`)

- ✅ Tâches par page : `free`, `data_config`, `exploration_chart_comment`,
  `estimation`, `results`, `robustness`
- 🔧 Contexte SCM injecté dans les prompts (structure présente, contenu à enrichir)
- ❌ Prompts `results` et `robustness` : injecter les valeurs numériques clés
  (RMSPE, p-valeur, ratio, poids dominants)
- ❌ Prompt `data_config` : détecter et signaler les risques (T₀ trop proche de T,
  trop peu de donneurs, NaN excessifs)

### 8.5 ContextBuilder (`IA_integration/context_builder.py`)

- 🔧 Construction du contexte SCM (données, config, résultats) en JSON/texte
- ❌ Limiter la taille du contexte injecté (max ~800 tokens) pour éviter le dépassement
  de fenêtre du modèle

### 8.6 Panneau IA (`app_ui/components/ai_panel.py`)

- ✅ Bannière de chargement du modèle
- ✅ Bouton désactivé si modèle non prêt
- ✅ Réponse persistée dans `st.session_state[f"ai_resp_{page_name}"]`
- ❌ Bouton "Effacer la réponse"
- ❌ Indicateur de durée d'inférence

---

## 9. Tests unitaires et validation scientifique

**Répertoire :** `tests/` (à créer / compléter)

### 9.1 Tests unitaires prioritaires

- ❌ `test_sc_format.py` : construction, invariants, edge cases (T0=2, J=1, K=0)
- ❌ `test_slsqp.py` : solution connue sur données simulées
- ❌ `test_random_search.py` : reproductibilité par graine, monotonicité de la loss
- ❌ `test_placebo.py` : vérifier p-valeur = 1/J si 0 placebo fait mieux
- ❌ `test_rmspe.py` : cas ratio=1 (absence d'effet), cas ratio >> 1
- ❌ `test_invalidation.py` : vérifier la cascade complète
- ❌ `test_data_prep.py` : panel irrégulier, NaN, covariable sans obs pré-traitement

### 9.2 🔬 Validation scientifique sur données de référence

> **Priorité absolue avant publication ou partage.**

- ❌ **Californie Proposition 99** (Abadie et al. 2010)
  - Reproduire les poids W* et V* publiés (Table 1 de l'article)
  - Vérifier gap 1989–2000 cohérent avec la Figure 1
  - Vérifier p-valeur placebo ≤ 0.05
- ❌ **Pays Basque** (Abadie & Gardeazabal 2003)
  - Reproduire les poids connus
  - Vérifier RMSPE_pré < 5% (niveau de référence)
- ⚠️ Tolérance numérique à définir : résultats attendus à ±1e-4 sur les poids

---

## 10. Extensions scientifiques (Phase 2)

> Non commencées. Planification préliminaire.

- ❌ **Test placebo en temps** : T₀ fictif balayé sur toute la période pré-traitement,
  garder le vrai T₀ comme dernier point
- ❌ **Intervalles de confiance** par bootstrap conformal
  (Chernozhukov et al., 2017 — placebo-based)
- ❌ **Augmented SCM** (Ben-Michael et al. 2021) :
  ajouter terme de correction par régression `sc_core/estimateur/methods/outer/augmented.py`
- ❌ **Synthetic DiD** (Arkhangelsky et al. 2021) :
  nouveau solveur avec poids temporels
- ❌ **Generalized SC** (Xu 2017) : facteur modèle interactif
- ❌ **Export rapport PDF** : générer un rapport structuré avec figures, tableaux,
  p-valeur et interprétation IA

---

## 11. Productionisation (Phase 3)

> Non commencée.

- ❌ **API REST** (`FastAPI`) exposant `/estimate`, `/robustness`, `/ai-comment`
- ❌ **Support GPU** : détecter CUDA / OpenVINO GPU et switcher automatiquement
- ❌ **Déploiement web** : Dockerfile, guide Streamlit Cloud / HuggingFace Spaces
- ❌ **Monitoring** : logger les erreurs d'estimation et d'inférence IA
- ❌ **Documentation API** : docstrings complètes + `sphinx` ou `mkdocs`
- ❌ **Documentation utilisateur** : guide pas-à-pas avec captures d'écran
- ❌ **Sécurité** : masquer chemins locaux et stacktraces dans l'UI de production

---

## Récapitulatif par priorité

### 🔴 Critique — Bloquerait une démonstration

| # | Tâche | Fichier |
|---|-------|---------|
| C1 | Lever exceptions typées sur invariants SCFormat | `sc_format.py` |
| C2 | Flag `estimation_stale` + invalidation sur changement de config | `invalidation.py`, `workflow.py` |
| C3 | Validation Californie Prop 99 + Pays Basque | `tests/` |
| C4 | LOO : gérer `|A| == 1` | `leave_one_out.py` |
| C5 | Backdating : stopper si < 2 pseudo-T₀ valides | `backdating.py` |

### 🟠 Haute priorité — Requis pour audit de code

| # | Tâche | Fichier |
|---|-------|---------|
| H1 | Tests unitaires SCFormat, SLSQP, RandomSearch | `tests/` |
| H2 | Tests unitaires placebo et RMSPE | `tests/` |
| H3 | Prompts `results` et `robustness` avec valeurs numériques | `prompt_builder.py` |
| H4 | Export figures (PNG) page Résultats | `4_Resultat.py` |
| H5 | Boutons d'exécution séparés par test de robustesse | `5_Robustesse.py` |
| H6 | Test de santé post-chargement du modèle IA | `model_loader.py` |

### 🟡 Moyenne priorité — Qualité et UX

| # | Tâche | Fichier |
|---|-------|---------|
| M1 | Avertissements sur qualité des données (NaN, T₀ marginal) | `1_Donnees.py` |
| M2 | Résumé numérique complet page Résultats | `4_Resultat.py` |
| M3 | Cache figures dans PlotManager | `plotting/manager.py` |
| M4 | Limiter taille contexte injecté dans prompts IA | `context_builder.py` |
| M5 | Statistiques descriptives pré/post page Exploration | `2_Exploration.py` |

---

## Notes de session

### Bugs corrigés (sessions précédentes)

| Bug | Correction | Session |
|-----|-----------|---------|
| `SCFormat` manquait `feature_names` | Champ ajouté à la dataclass | 2 |
| `select_valid_predictor_vars` rejetait covariables avec < T₀ obs | Accepte ≥ 1 obs | 2 |
| `get_backend_outcome_and_predictors` retournait `[]` au lieu de `None` | Correction retour | 2 |
| `st.columns(3)` → erreur d'unpacking sur 2 variables | Corrigé en `st.columns(2)` | 2 |
| Covariables irrégulières (ex : recensement décennal) rejetées | Acceptées nativement | 2 |
| `sc_core/IA_integration/codes/manager.py` sans `generate_text()` | Méthode ajoutée | 3 |
| Pas de singleton pour le modèle IA → rechargement à chaque page | `@st.cache_resource` | 3 |

### Points d'architecture à ne pas casser

- `sc_core` **ne doit jamais** importer `streamlit`
- Toutes les clés `session_state` passent **exclusivement** par `keys.py`
- Le `SCFormat` est le **seul** contrat de données entre UI et core
- Le chargement du modèle IA est un **singleton** par session Streamlit
