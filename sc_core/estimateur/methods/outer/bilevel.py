

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from ...core.types import EstimationResult
from ...core.status import SolverStatus
from ...core.callbacks import OptimizationCallback
from ...core.exceptions import SolverFailureError
from ...utils.grouping import build_group_index, expand_group_weights
from ...utils.scaling import safe_softmax
from ...utils.history import new_history, append_history
from ...validation.scoring import mspe
from ..inner.quadratic import build_qp_from_xv


def fit_via_bilevel_scipy(
    *,
    X1: np.ndarray,
    X0: np.ndarray,
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    row_var,
    group_names,
    donor_names,
    inner_solver,
    n_restarts: int = 5,
    maxiter: int = 200,
    seed: int = 123,
    method: str = "Powell",
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
    global_iter = 0

    def evaluate_theta(theta: np.ndarray):
        nonlocal global_iter, best_loss, best_w, best_Vvar, best_Vdiag

        global_iter += 1

        Vvar = safe_softmax(theta)
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
            iteration=global_iter,
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
                method="bilevel",
                iteration=global_iter,
                n_iterations_total=None,
                status_message=f"Bilevel outer iteration {global_iter}",
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
                    "solver": "bilevel",
                    "iteration": global_iter,
                },
            )

        return float(loss)

    last_res = None

    for _ in range(int(n_restarts)):
        theta0 = rng.normal(size=G)

        res = minimize(
            fun=evaluate_theta,
            x0=theta0,
            method=str(method),
            options={"maxiter": int(maxiter), "disp": False},
        )
        last_res = res

    if best_w is None:
        raise SolverFailureError("Bilevel outer optimization failed to find a feasible solution.")

    message = "Bilevel outer optimization completed successfully."
    success = True
    status = SolverStatus.CONVERGED

    if last_res is not None and not last_res.success:
        message = f"Last outer restart ended with message: {last_res.message}"

    return EstimationResult(
        w=best_w,
        donor_names=list(donor_names),
        Vvar=best_Vvar,
        group_names=list(group_names),
        Vdiag=best_Vdiag,
        loss=float(best_loss),
        success=success,
        status=status,
        message=message,
        n_iter=int(global_iter),
        history=history,
    )