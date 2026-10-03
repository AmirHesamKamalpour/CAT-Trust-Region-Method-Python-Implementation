from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from optimizers.types import Array, CATParams, EvalCounters
from utils.linalg import matvec, norm2


class ShiftedSystemSolver:
    """Solve ``(H + delta I)x = b`` and test positive definiteness."""

    def __init__(
        self,
        n: int,
        counters: EvalCounters,
        params: CATParams,
        H_mat: Optional[sp.spmatrix] = None,
        H_op: Optional[spla.LinearOperator] = None,
    ):
        self.n = n
        self.counters = counters
        self.params = params

        self.H_mat: Optional[sp.csc_matrix] = None
        self.H_op: spla.LinearOperator

        if H_mat is not None:
            Hc = H_mat.tocsc()
            Hc = ((Hc + Hc.T) * 0.5).tocsc()
            Hc.eliminate_zeros()

            self.H_mat = Hc
            self.n = Hc.shape[0]
            self.H_op = H_op if H_op is not None else spla.aslinearoperator(Hc)
            self._base_diag = self.H_mat.diagonal().copy()
        else:
            if H_op is None:
                raise ValueError("Provide either H_mat (sparse) or H_op (LinearOperator).")
            self.H_op = H_op

        self._cholmod_available = False
        self._cholmod = None
        try:
            from sksparse.cholmod import cholesky  # type: ignore

            self._cholmod_available = self.H_mat is not None
            self._cholmod = cholesky
        except Exception:
            self._cholmod_available = False

        self._last_delta: Optional[float] = None
        self._last_factor = None
        self._last_lu = None
        self._last_shifted: Optional[sp.csc_matrix] = None
        self._pd_delta_min: Optional[float] = None

    def _shifted_matrix(self, delta: float) -> sp.csc_matrix:
        assert self.H_mat is not None

        if delta == 0.0:
            return self.H_mat

        if self._last_delta == delta and self._last_shifted is not None:
            return self._last_shifted

        A = self.H_mat.copy()
        A.setdiag(self._base_diag + float(delta))
        A.eliminate_zeros()
        self._last_shifted = A
        return A

    def estimate_pd_delta_min(self) -> float:
        """Estimate the minimum shift making ``H + delta I`` positive definite."""

        if self._pd_delta_min is not None:
            return self._pd_delta_min

        try:
            lam_min = float(
                spla.eigsh(
                    self.H_op,
                    k=1,
                    which="SA",
                    return_eigenvectors=False,
                    tol=self.params.eig_tol,
                    maxiter=self.params.eig_maxiter,
                )[0]
            )
        except Exception:
            v = np.random.randn(self.n)
            v /= norm2(v)
            lam_min = 0.0
            for _ in range(30):
                w = -matvec(self.H_op, v)
                self.counters.Hv += 1
                nw = norm2(w)
                if nw == 0:
                    lam_min = 0.0
                    break
                v = w / nw
                lam_min = -float(v @ matvec(self.H_op, v))
                self.counters.Hv += 1

        self._pd_delta_min = float(max(0.0, -lam_min + self.params.pd_jitter))
        return self._pd_delta_min

    def is_pos_def(self, delta: float) -> bool:
        if delta < 0:
            return False

        if self._cholmod_available and self.H_mat is not None:
            try:
                _ = self._cholmod(self._shifted_matrix(delta))
                self.counters.fact += 1
                return True
            except Exception:
                return False

        return delta >= self.estimate_pd_delta_min()

    def solve(self, b: Array, delta: float) -> Array:
        """Solve ``(H + delta I)x = b``."""

        if delta < 0:
            raise ValueError("delta must be >= 0")

        if self.H_mat is not None:
            A = self._shifted_matrix(delta)

            if self._cholmod_available and self._cholmod is not None:
                if self._last_factor is None or self._last_delta != delta:
                    self._last_factor = self._cholmod(A)
                    self._last_delta = delta
                    self.counters.fact += 1
                return np.asarray(self._last_factor(b)).reshape(-1)

            if self._last_lu is None or self._last_delta != delta:
                self._last_lu = spla.splu(A)
                self._last_delta = delta
                self.counters.fact += 1
            return np.asarray(self._last_lu.solve(b)).reshape(-1)

        n = self.n
        Aop = spla.LinearOperator(
            (n, n),
            matvec=lambda v: matvec(self.H_op, v) + delta * v,
            dtype=float,
        )

        def counted_matvec(v: Array) -> Array:
            self.counters.Hv += 1
            return Aop.matvec(v)

        Aop_counted = spla.LinearOperator((n, n), matvec=counted_matvec, dtype=float)
        try:
            x, info = spla.minres(
                Aop_counted,
                b,
                rtol=self.params.lin_solve_tol,
                maxiter=self.params.lin_solve_maxit,
            )
        except TypeError:
            # Compatibility with older SciPy releases.
            x, info = spla.minres(
                Aop_counted,
                b,
                tol=self.params.lin_solve_tol,
                maxiter=self.params.lin_solve_maxit,
            )
        if info != 0:
            raise RuntimeError(f"minres failed (info={info})")
        return np.asarray(x).reshape(-1)
