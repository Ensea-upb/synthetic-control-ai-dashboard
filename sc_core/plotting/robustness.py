from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .base import create_figure_ax, finalize_ax, add_treatment_line, finalize_legend


# ============================================================
# PLACEBO GAPS
# ============================================================

def plot_placebo_gaps(
    time_index,
    gaps_dict: Dict[str, np.ndarray],
    treated_unit: str,
    base_gap: Optional[np.ndarray] = None,
    T0=None,
):
    time_arr = np.asarray(time_index)
    fig, ax = create_figure_ax((13, 7))

    for unit, gap_arr in gaps_dict.items():
        gap_arr = np.asarray(gap_arr, dtype=float)
        if len(gap_arr) != len(time_arr):
            continue
        ax.plot(time_arr, gap_arr, color="grey", linewidth=1, alpha=0.35, zorder=1)

    if base_gap is not None:
        base_gap = np.asarray(base_gap, dtype=float)
        if len(base_gap) == len(time_arr):
            ax.plot(time_arr, base_gap, color="#e63946", linewidth=3,
                    label=f"{treated_unit} (traité)", zorder=3)

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    if T0 is not None:
        add_treatment_line(ax, T0, label=f"T0 = {T0}")

    n_placebos = len(gaps_dict)
    ax.set_title(
        f"Test placebo (espace) — {n_placebos} unités placebo + traité en rouge",
        fontsize=13, fontweight="bold",
    )
    finalize_ax(ax, xlabel="Période", ylabel="Gap (traité − synthétique)")
    finalize_legend(ax)
    fig.tight_layout()
    return fig


# ============================================================
# RMSPE DISTRIBUTION
# ============================================================

def plot_rmspe_distribution(rmspe_info):
    if isinstance(rmspe_info, pd.Series):
        fig, ax = create_figure_ax()
        rmspe_info.plot(kind="bar", ax=ax, color="steelblue")
        finalize_ax(ax, "RMSPE ratio", "Unité", "Ratio")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        return fig

    if not isinstance(rmspe_info, dict) or len(rmspe_info) == 0:
        fig, ax = create_figure_ax()
        ax.text(0.5, 0.5, "Aucune donnée RMSPE disponible",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    units = list(rmspe_info.keys())
    pre_vals = []
    post_vals = []
    for u in units:
        entry = rmspe_info[u]
        if isinstance(entry, dict):
            pre_vals.append(float(entry.get("pre_rmspe", 0) or 0))
            post_vals.append(float(entry.get("post_rmspe", 0) or 0))
        else:
            pre_vals.append(0.0)
            post_vals.append(float(entry) if entry else 0.0)

    x = np.arange(len(units))
    width = 0.4
    fig, ax = create_figure_ax((max(10, len(units) * 0.7 + 2), 6))
    ax.bar(x - width / 2, pre_vals, width, label="RMSPE pré-traitement", color="#457b9d", alpha=0.85)
    ax.bar(x + width / 2, post_vals, width, label="RMSPE post-traitement", color="#e63946", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(units, rotation=45, ha="right")
    ax.set_title("Distribution des RMSPE pré / post par unité", fontsize=13, fontweight="bold")
    finalize_ax(ax, xlabel="Unité", ylabel="RMSPE")
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# RMSPE RATIO RANKING
# ============================================================

def plot_rmspe_ratio_ranking(ratio_series: pd.Series, treated: Optional[str] = None):
    s = ratio_series.sort_values(ascending=False)
    fig, ax = create_figure_ax((max(10, len(s) * 0.7 + 2), 6))
    colors = ["#e63946" if i == treated else "#457b9d" for i in s.index]
    ax.bar(range(len(s)), s.values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index.tolist(), rotation=45, ha="right")
    ax.set_title("Classement RMSPE post/pré — unité traitée en rouge", fontsize=13, fontweight="bold")
    finalize_ax(ax, xlabel="Unité", ylabel="Ratio RMSPE post / pré")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Ratio = 1")
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# LEAVE-ONE-OUT
# ============================================================

def plot_leave_one_out_gaps(
    time_index,
    base_gap: np.ndarray,
    gaps_by_donor: Dict[str, np.ndarray],
    treated_label: str = "Baseline (tous donneurs)",
    T0=None,
):
    time_arr = np.asarray(time_index)
    base_gap = np.asarray(base_gap, dtype=float)
    n_donors = len(gaps_by_donor)
    cmap = cm.get_cmap("tab10", max(n_donors, 1))

    fig, ax = create_figure_ax((13, 7))
    for i, (donor, gap_arr) in enumerate(gaps_by_donor.items()):
        gap_arr = np.asarray(gap_arr, dtype=float)
        if len(gap_arr) != len(time_arr):
            continue
        ax.plot(time_arr, gap_arr, color=cmap(i), linewidth=1.5,
                alpha=0.6, label=f"Sans {donor}", zorder=2)

    if len(base_gap) == len(time_arr):
        ax.plot(time_arr, base_gap, color="#e63946", linewidth=3,
                label=treated_label, zorder=3)

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    if T0 is not None:
        add_treatment_line(ax, T0, label=f"T0 = {T0}")

    ax.set_title(
        f"Leave-one-out — impact du retrait de chaque donneur ({n_donors} runs)",
        fontsize=13, fontweight="bold",
    )
    finalize_ax(ax, xlabel="Période", ylabel="Gap (traité − synthétique)")
    finalize_legend(ax, outside=n_donors > 8)
    fig.tight_layout()
    return fig


# ============================================================
# BACKDATING
# ============================================================

def plot_backdating_gaps(time_index, runs: List[Any], real_T0=None):
    time_arr = np.asarray(time_index)
    n_runs = len(runs)

    if n_runs == 0:
        fig, ax = create_figure_ax()
        ax.text(0.5, 0.5, "Aucun run backdating disponible",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    cmap = cm.get_cmap("viridis", n_runs)
    fig, ax = create_figure_ax((13, 7))

    for i, run in enumerate(runs):
        gaps = np.asarray(run.gaps, dtype=float)
        pseudo_t0 = run.pseudo_t0
        if len(gaps) != len(time_arr):
            continue
        is_real = (real_T0 is not None and pseudo_t0 == real_T0)
        ax.plot(time_arr, gaps, color=cmap(i),
                linewidth=3 if is_real else 1.5,
                alpha=1.0 if is_real else 0.65,
                label=f"pseudo-T0 = {pseudo_t0}" + (" ★" if is_real else ""),
                zorder=3 if is_real else 2)
        ax.axvline(pseudo_t0, color=cmap(i), linestyle=":", linewidth=0.8, alpha=0.4)

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    if real_T0 is not None:
        add_treatment_line(ax, real_T0, label=f"T0 réel = {real_T0}")

    ax.set_title(f"Backdating — gaps sous {n_runs} pseudo-T0 différents",
                 fontsize=13, fontweight="bold")
    finalize_ax(ax, xlabel="Période", ylabel="Gap (traité − synthétique)")
    finalize_legend(ax, outside=n_runs > 8)
    fig.tight_layout()
    return fig


def plot_backdating_ratio_bars(runs: List[Any]):
    labels, ratios = [], []
    for run in runs:
        gaps = np.asarray(run.gaps, dtype=float)
        n_pre = len(run.pre_periods_used)
        if n_pre == 0 or len(gaps) < n_pre + 1:
            continue
        pre_rmspe = float(np.sqrt(np.mean(gaps[:n_pre] ** 2)))
        post_rmspe = float(np.sqrt(np.mean(gaps[n_pre:] ** 2))) if len(gaps) > n_pre else 0.0
        ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")
        labels.append(str(run.pseudo_t0))
        ratios.append(ratio)

    if not labels:
        fig, ax = create_figure_ax()
        ax.text(0.5, 0.5, "Impossible de calculer les ratios",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = create_figure_ax((max(8, len(labels) * 0.8 + 2), 5))
    ax.bar(labels, ratios, color="#1d3557", alpha=0.85)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(1.0, color="#e63946", linestyle="--", linewidth=1.5, label="Ratio = 1")
    ax.set_title("Backdating — ratio RMSPE post/pré par pseudo-T0",
                 fontsize=12, fontweight="bold")
    finalize_ax(ax, xlabel="Pseudo-T0", ylabel="Ratio RMSPE post/pré")
    ax.legend()
    fig.tight_layout()
    return fig
