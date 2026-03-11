from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ...core.status import SolverStatus
from ...core.types import InnerSolveResult
from ...core.exceptions import ShapeMismatchError, SolverFailureError
from ...utils.simplex import clean_simplex_weights, simplex_bounds


class SLSQPInnerSolver:
    name = "slsqp"

    def __init__(
        self,
        maxiter: int = 1000,
        ftol: float = 1e-10,
        ridge: float = 1e-10,
        fallback_method: str = "trust-constr",
    ):
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.ridge = float(ridge)
        self.fallback_method = str(fallback_method)

    def _validate_inputs(self, Q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = np.asarray(Q, dtype=float)
        p = np.asarray(p, dtype=float).reshape(-1)

        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ShapeMismatchError("Q must be square.")
        if p.shape[0] != Q.shape[0]:
            raise ShapeMismatchError("p must have the same dimension as Q.")

        if not np.all(np.isfinite(Q)):
            raise SolverFailureError("Inner solver received non-finite values in Q.")
        if not np.all(np.isfinite(p)):
            raise SolverFailureError("Inner solver received non-finite values in p.")

        return Q, p

    def _prepare_problem(self, Q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q, p = self._validate_inputs(Q, p)

        # Symmetrize Q
        Q = 0.5 * (Q + Q.T)

        # Light ridge regularization for numerical stability
        if self.ridge > 0.0:
            Q = Q + self.ridge * np.eye(Q.shape[0], dtype=float)

        return Q, p

    def _build_objective(self, Q: np.ndarray, p: np.ndarray):
        def fun(w: np.ndarray) -> float:
            val = float(0.5 * w @ Q @ w + p @ w)
            if not np.isfinite(val):
                return 1e20
            return val

        def jac(w: np.ndarray) -> np.ndarray:
            g = Q @ w + p
            if not np.all(np.isfinite(g)):
                return np.full_like(w, 1e10, dtype=float)
            return g

        return fun, jac

    def _run_slsqp(self, fun, jac, x0: np.ndarray, J: int):
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones_like(w),
            }
        ]

        return minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            bounds=simplex_bounds(J),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.maxiter, "ftol": self.ftol, "disp": False},
        )

    def _run_fallback(self, fun, jac, x0: np.ndarray, J: int):
        if self.fallback_method != "trust-constr":
            return None

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones_like(w),
            }
        ]

        try:
            res = minimize(
                fun=fun,
                x0=x0,
                jac=jac,
                bounds=simplex_bounds(J),
                constraints=constraints,
                method="trust-constr",
                options={"maxiter": self.maxiter, "verbose": 0},
            )
            return res
        except Exception:
            return None

    def solve(self, Q: np.ndarray, p: np.ndarray) -> InnerSolveResult:
        Q, p = self._prepare_problem(Q, p)

        J = Q.shape[0]
        x0 = np.ones(J, dtype=float) / float(J)

        fun, jac = self._build_objective(Q, p)

        res = self._run_slsqp(fun, jac, x0, J)
        solver_used = "SLSQP"

        if (not res.success) or (not np.all(np.isfinite(res.x))) or (not np.isfinite(res.fun)):
            res_fb = self._run_fallback(fun, jac, x0, J)
            if res_fb is not None and res_fb.success and np.all(np.isfinite(res_fb.x)) and np.isfinite(res_fb.fun):
                res = res_fb
                solver_used = self.fallback_method.upper()

        if (not res.success) or (not np.all(np.isfinite(res.x))):
            raise SolverFailureError(
                f"Inner W optimization failed with {solver_used}. "
                f"message={res.message}; dim={J}; ridge={self.ridge}"
            )

        w = clean_simplex_weights(res.x)
        obj = fun(w)

        if not np.isfinite(obj):
            raise SolverFailureError(
                f"Inner W optimization produced a non-finite objective after cleaning. "
                f"solver={solver_used}; dim={J}; ridge={self.ridge}"
            )

        return InnerSolveResult(
            weights=w,
            objective_value=float(obj),
            success=True,
            status=SolverStatus.CONVERGED,
            message=f"{solver_used}: {res.message}",
            n_iter=int(getattr(res, "nit", 0)),
            history=[],
        )