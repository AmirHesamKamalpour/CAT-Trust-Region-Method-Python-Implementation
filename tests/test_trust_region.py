import numpy as np
import scipy.sparse as sp

from optimizers.trust_region import TrustRegionSubproblemSolver
from optimizers.types import CATParams, EvalCounters


def test_trust_region_returns_newton_step_when_feasible():
    H = sp.eye(3, format="csc")
    g = np.array([0.1, -0.2, 0.3])
    solver = TrustRegionSubproblemSolver(CATParams(), EvalCounters())

    d, delta, stats = solver.solve(
        g=g,
        H=H,
        r=10.0,
        eps_k=np.linalg.norm(g),
        delta_prev=0.0,
    )

    np.testing.assert_allclose(d, -g, rtol=1e-8, atol=1e-10)
    assert delta == 0.0
    assert stats["mode"] == "newton"
