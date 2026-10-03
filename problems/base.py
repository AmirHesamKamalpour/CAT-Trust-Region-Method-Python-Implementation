from __future__ import annotations

from typing import Protocol, Union

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from optimizers.types import Array

SparseMatrix = sp.spmatrix
Hessian = Union[SparseMatrix, spla.LinearOperator]


class UnconstrainedProblem(Protocol):
    """Minimal interface required by the CAT optimizer."""

    def obj(self, x: Array) -> float: ...

    def grad(self, x: Array) -> Array: ...

    def hess(self, x: Array) -> Hessian: ...

    def x0(self) -> Array: ...
