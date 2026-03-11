

class EstimatorError(Exception):
    """Base exception for the estimateur package."""


class ShapeMismatchError(EstimatorError):
    """Raised when input shapes are inconsistent."""


class InvalidSimplexError(EstimatorError):
    """Raised when simplex constraints are violated."""


class SolverFailureError(EstimatorError):
    """Raised when an optimization solver fails."""


class InvalidConfigError(EstimatorError):
    """Raised when a solver configuration is invalid."""


class DataPreparationError(EstimatorError):
    """Raised when raw tabular data cannot be converted into solver matrices."""