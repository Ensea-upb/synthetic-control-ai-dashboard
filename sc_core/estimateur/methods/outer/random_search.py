from __future__ import annotations

import time
import numpy as np

from ...core.types import EstimationResult
from ...core.status import SolverStatus
from ...core.callbacks import OptimizationCallback
from ...utils.grouping import build_group_index, expand_group_weights
from ...utils.history import new_history, append_history
from ..inner.quadratic import build_qp_from_xv


def mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(err ** 2))


def fit_via_random_search(
    *,
    X1: np.ndarray,
    X0: np.ndarray,
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    row_var,
    group_names,
    donor_names,
    inner_solver,
    n_iter: int = 200,
    seed: int = 123,
    callback: OptimizationCallback | None = None,
) -> EstimationResult:
    rng = np.random.default_rng(int(seed))
    group_idx = build_group_index(row_var=row_var, group_names=group_names)
    G = len(group_names)

    best_loss = float("inf")
    best_w = None
    best_Vvar = None
    best_Vdiag = None

    history = new_history()
    t0 = time.perf_counter()

    for it in range(1, int(n_iter) + 1):
        Vvar = rng.dirichlet(np.ones(G, dtype=float))
        Vdiag = expand_group_weights(Vvar, group_idx)

        Q, p = build_qp_from_xv(X1=X1, X0=X0, Vdiag=Vdiag)
        inner_res = inner_solver.solve(Q, p)
        w = inner_res.weights

        y_synth_pre = Y0_pre @ w
        loss = mspe(Y1_pre, y_synth_pre)

        if loss < best_loss:
            best_loss = float(loss)
            best_w = w.copy()
            best_Vvar = Vvar.copy()
            best_Vdiag = Vdiag.copy()

        append_history(
            history,
            stage="outer",
            iteration=it,
            loss_current=loss,
            loss_best=best_loss,
            elapsed_sec=time.perf_counter() - t0,
            payload={
                "w_current": w.copy(),
                "Vvar_current": Vvar.copy(),
                "Vdiag_current": Vdiag.copy(),
            },
        )

        if callback is not None:
            callback(
                stage="outer",
                method="random_search",
                iteration=it,
                n_iterations_total=int(n_iter),
                status_message=f"Random search iteration {it}/{int(n_iter)}",
                objective_value=float(loss),
                best_objective_value=float(best_loss),
                loss_current=float(loss),
                loss_best=float(best_loss),
                w_current=w.copy(),
                weights=w.copy(),
                Vvar_current=Vvar.copy(),
                Vdiag_current=Vdiag.copy(),
                covariate_weights=Vdiag.copy(),
                raw_snapshot={
                    "solver": "random_search",
                    "iteration": it,
                },
            )

    if best_w is None:
        return EstimationResult(
            w=np.array([]),
            donor_names=list(donor_names),
            Vvar=np.array([]),
            group_names=list(group_names),
            Vdiag=np.array([]),
            loss=float("inf"),
            success=False,
            status=SolverStatus.FAILED,
            message="Random search failed to produce a feasible solution.",
            n_iter=int(n_iter),
            history=history,
        )

    return EstimationResult(
        w=best_w,
        donor_names=list(donor_names),
        Vvar=best_Vvar,
        group_names=list(group_names),
        Vdiag=best_Vdiag,
        loss=float(best_loss),
        success=True,
        status=SolverStatus.CONVERGED,
        message="Random search completed successfully.",
        n_iter=int(n_iter),
        history=history,
    )