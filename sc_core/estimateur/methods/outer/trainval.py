from __future__ import annotations

import time
import numpy as np

from ...core.types import EstimationResult
from ...core.status import SolverStatus
from ...core.callbacks import OptimizationCallback
from ...utils.grouping import build_group_index, expand_group_weights
from ...utils.history import new_history, append_history
from ...validation.scoring import mspe
from ..inner.quadratic import build_qp_from_xv


def fit_via_trainval_random_search(
    *,
    X1_train: np.ndarray,
    X0_train: np.ndarray,
    row_var_train,
    X1_full: np.ndarray,
    X0_full: np.ndarray,
    row_var_full,
    group_names,
    donor_names,
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    inner_solver,
    n_iter: int = 300,
    seed: int = 123,
    callback: OptimizationCallback | None = None,
) -> EstimationResult:
    rng = np.random.default_rng(int(seed))

    Y1_pre = np.asarray(Y1_pre, dtype=float).reshape(-1)
    Y0_pre = np.asarray(Y0_pre, dtype=float)

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    if X1_train.shape[0] == 0:
        return EstimationResult(
            w=np.array([]),
            donor_names=list(donor_names),
            Vvar=np.array([]),
            group_names=list(group_names),
            Vdiag=np.array([]),
            loss=float("inf"),
            success=False,
            status=SolverStatus.FAILED,
            message="Train X design is empty. Cannot run train/validation selection.",
            n_iter=0,
            history=[],
        )

    G = len(group_names)
    group_idx_train = build_group_index(row_var=row_var_train, group_names=group_names)
    group_idx_full = build_group_index(row_var=row_var_full, group_names=group_names)

    best_loss_val = float("inf")
    best_Vvar = None

    history = new_history()
    t0 = time.perf_counter()

    # -----------------------------------------------------
    # 1) Select V on train/validation
    # -----------------------------------------------------
    for it in range(1, int(n_iter) + 1):
        Vvar = rng.dirichlet(np.ones(G, dtype=float))

        Vdiag_train = expand_group_weights(Vvar, group_idx_train)
        Q_train, p_train = build_qp_from_xv(X1=X1_train, X0=X0_train, Vdiag=Vdiag_train)
        inner_res_train = inner_solver.solve(Q_train, p_train)
        w_train = inner_res_train.weights

        y_synth_pre_trainfit = Y0_pre @ w_train
        loss_train = mspe(Y1_pre[train_idx], y_synth_pre_trainfit[train_idx])
        loss_val = mspe(Y1_pre[val_idx], y_synth_pre_trainfit[val_idx])

        if loss_val < best_loss_val:
            best_loss_val = float(loss_val)
            best_Vvar = Vvar.copy()

        append_history(
            history,
            stage="outer",
            iteration=it,
            loss_current=loss_val,
            loss_best=best_loss_val,
            elapsed_sec=time.perf_counter() - t0,
            payload={
                "loss_train": float(loss_train),
                "loss_val": float(loss_val),
                "w_current": w_train.copy(),
                "Vvar_current": Vvar.copy(),
            },
        )

        if callback is not None:
            callback(
                stage="outer",
                method="trainval",
                iteration=it,
                n_iterations_total=int(n_iter),
                status_message=f"Train/validation search iteration {it}/{int(n_iter)}",
                objective_value=float(loss_val),
                best_objective_value=float(best_loss_val),
                loss_current=float(loss_val),
                loss_best=float(best_loss_val),
                w_current=w_train.copy(),
                weights=w_train.copy(),
                Vvar_current=Vvar.copy(),
                Vdiag_current=Vdiag_train.copy(),
                covariate_weights=Vdiag_train.copy(),
                raw_snapshot={
                    "solver": "trainval",
                    "iteration": it,
                    "loss_train": float(loss_train),
                    "loss_val": float(loss_val),
                },
            )

    if best_Vvar is None:
        return EstimationResult(
            w=np.array([]),
            donor_names=list(donor_names),
            Vvar=np.array([]),
            group_names=list(group_names),
            Vdiag=np.array([]),
            loss=float("inf"),
            success=False,
            status=SolverStatus.FAILED,
            message="Train/validation search failed to select a feasible V.",
            n_iter=int(n_iter),
            history=history,
        )

    # -----------------------------------------------------
    # 2) Refit final W on full pre-treatment using V*
    # -----------------------------------------------------
    Vdiag_full = expand_group_weights(best_Vvar, group_idx_full)
    Q_full, p_full = build_qp_from_xv(X1=X1_full, X0=X0_full, Vdiag=Vdiag_full)
    inner_res_full = inner_solver.solve(Q_full, p_full)
    w_final = inner_res_full.weights

    return EstimationResult(
        w=w_final,
        donor_names=list(donor_names),
        Vvar=best_Vvar,
        group_names=list(group_names),
        Vdiag=Vdiag_full,
        loss=float(best_loss_val),
        success=True,
        status=SolverStatus.CONVERGED,
        message="Train/validation selection completed successfully. Final W refit on full pre-treatment.",
        n_iter=int(n_iter),
        history=history,
    )