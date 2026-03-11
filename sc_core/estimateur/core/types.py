from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np

from .status import SolverStatus


@dataclass
class IterationRecord:
    stage: str
    iteration: int
    loss_current: float
    loss_best: float
    elapsed_sec: float | None = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedMatrices:
    X1: np.ndarray
    X0: np.ndarray
    Y1_pre: np.ndarray
    Y0_pre: np.ndarray
    row_var: List[str]
    row_time: List[Any]
    group_names: List[str]
    donor_names: List[str]
    pre_periods: List[Any]


@dataclass
class InnerSolveResult:
    weights: np.ndarray
    objective_value: float
    success: bool
    status: SolverStatus
    message: str
    n_iter: int
    history: List[IterationRecord] = field(default_factory=list)


@dataclass
class EstimationResult:
    w: np.ndarray
    donor_names: List[str]
    Vvar: np.ndarray
    group_names: List[str]
    Vdiag: np.ndarray
    loss: float
    success: bool
    status: SolverStatus
    message: str
    n_iter: int
    history: List[IterationRecord] = field(default_factory=list)

    @property
    def donor_weights_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.donor_names, self.w)}

    @property
    def covariate_weights_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.group_names, self.Vvar)}

    @property
    def objective_history(self) -> List[float]:
        return [float(h.loss_current) for h in self.history if h.stage == "outer"]

    @property
    def best_objective_history(self) -> List[float]:
        return [float(h.loss_best) for h in self.history if h.stage == "outer"]