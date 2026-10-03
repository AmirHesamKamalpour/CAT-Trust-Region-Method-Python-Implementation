import numpy as np
import scipy.sparse as sp

from optimizers import CATOptimizer, CATParams


class QuadraticProblem:
    def __init__(self):
        self._x0 = np.array([1.0, -2.0, 0.5])

    def x0(self):
        return self._x0.copy()

    def obj(self, x):
        return 0.5 * float(x @ x)

    def grad(self, x):
        return np.asarray(x, dtype=float)

    def hess(self, x):
        return sp.eye(x.size, format="csc")


def test_cat_converges_on_identity_quadratic():
    optimizer = CATOptimizer(CATParams(tol=1e-8, max_iter=50))
    result = optimizer.minimize(QuadraticProblem())

    assert result.status == "SUCCESS"
    assert result.eps <= 1e-8
    np.testing.assert_allclose(result.x, np.zeros(3), atol=1e-8)
