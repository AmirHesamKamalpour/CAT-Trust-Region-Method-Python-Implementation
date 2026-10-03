import numpy as np
import scipy.sparse as sp

from problems import PyCUTEstProblem


class DummyCUTEstProblem:
    x0 = np.array([1.0, 2.0])

    def obj(self, x):
        return float(x @ x)

    def grad(self, x):
        return 2.0 * x

    def hess(self, x):
        return np.array([[2.0, 0.0], [0.0, 2.0]])


def test_cutest_adapter_converts_dense_hessian_to_sparse():
    problem = PyCUTEstProblem(DummyCUTEstProblem())
    H = problem.hess(problem.x0())

    assert sp.issparse(H)
    np.testing.assert_allclose(H.toarray(), 2.0 * np.eye(2))
