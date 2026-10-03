import numpy as np
import scipy.sparse as sp

from optimizers.linear_solver import ShiftedSystemSolver
from optimizers.types import CATParams, EvalCounters


def test_shifted_system_solver_matches_dense_solution():
    H = sp.csc_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])
    delta = 0.25

    solver = ShiftedSystemSolver(
        n=2,
        counters=EvalCounters(),
        params=CATParams(),
        H_mat=H,
    )
    x = solver.solve(b, delta)

    expected = np.linalg.solve(H.toarray() + delta * np.eye(2), b)
    np.testing.assert_allclose(x, expected, rtol=1e-9, atol=1e-11)
