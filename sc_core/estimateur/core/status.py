

from enum import Enum


class SolverStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"