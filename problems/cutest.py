from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from optimizers.types import Array
from problems.base import SparseMatrix


class PyCUTEstProblem:
    """Adapter exposing a pycutest problem through ``UnconstrainedProblem``."""

    def __init__(self, problem: Any):
        self.problem = problem
        self._x0 = np.asarray(getattr(problem, "x0"), dtype=float).reshape(-1)

    def x0(self) -> Array:
        return self._x0.copy()

    def obj(self, x: Array) -> float:
        return float(self.problem.obj(x))

    def grad(self, x: Array) -> Array:
        return np.asarray(self.problem.grad(x), dtype=float).reshape(-1)

    def hess(self, x: Array) -> SparseMatrix:
        H = self.problem.hess(x)
        if sp.issparse(H):
            return H.tocsc()
        return sp.csc_matrix(np.asarray(H, dtype=float))
