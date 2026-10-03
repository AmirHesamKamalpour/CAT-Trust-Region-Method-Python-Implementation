from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from optimizers.types import Array, CATParams, EvalCounters
from problems.base import Hessian


def norm2(v: Array) -> float:
    return float(np.linalg.norm(v))


def as_linear_operator(H: Hessian, n: int) -> spla.LinearOperator:
    if isinstance(H, spla.LinearOperator):
        return H
    return spla.aslinearoperator(H)


def matvec(H: Hessian, v: Array) -> Array:
    if isinstance(H, spla.LinearOperator):
        return np.asarray(H.matvec(v), dtype=float).reshape(-1)
    return np.asarray(H @ v, dtype=float).reshape(-1)


def quadratic_model(g: Array, H: Hessian, d: Array) -> float:
    Hd = matvec(H, d)
    return 0.5 * float(d @ Hd) + float(g @ d)


def residual_stationarity(g: Array, H: Hessian, d: Array, delta: float) -> float:
    Hd = matvec(H, d)
    return norm2(Hd + g + delta * d)


def check_trs_conditions(
    g: Array,
    H: Hessian,
    d: Array,
    delta: float,
    r: float,
    eps_k: float,
    params: CATParams,
) -> Tuple[bool, Dict[str, float]]:
    """Check the trust-region termination conditions (6a)-(6d)."""

    nd = norm2(d)
    stat = residual_stationarity(g, H, d, delta)

    eps = 1e-12 * max(1.0, r, eps_k)
    c6a = stat <= params.gamma1 * eps_k + eps
    c6b = True if delta == 0.0 else nd >= params.gamma2 * r - eps
    c6c = nd <= r + eps

    Mk = quadratic_model(g, H, d)
    rhs = -params.gamma3 * (delta / 2.0) * (nd**2)
    c6d = Mk <= rhs + eps

    ok = c6a and c6b and c6c and c6d
    info = {
        "||d||": nd,
        "delta": float(delta),
        "stationarity": stat,
        "Mk": float(Mk),
        "rhs_6d": float(rhs),
        "c6a": float(c6a),
        "c6b": float(c6b),
        "c6c": float(c6c),
        "c6d": float(c6d),
    }
    return ok, info


def estimate_spectral_norm(
    H: Hessian,
    n: int,
    counters: EvalCounters,
    params: CATParams,
) -> float:
    """Estimate ``||H||_2`` for the CAT initial-radius heuristic."""

    H_op = as_linear_operator(H, n)
    try:
        lam = float(
            spla.eigsh(
                H_op,
                k=1,
                which="LM",
                return_eigenvectors=False,
                tol=max(params.eig_tol, 1e-4),
                maxiter=params.eig_maxiter,
            )[0]
        )
        return abs(lam)
    except Exception:
        v = np.random.randn(n)
        v /= max(norm2(v), 1e-300)
        lam = 0.0
        for _ in range(30):
            w = matvec(H_op, v)
            counters.Hv += 1
            nw = norm2(w)
            if nw == 0:
                break
            v = w / nw
            lam = float(v @ matvec(H_op, v))
            counters.Hv += 1
        return abs(lam)
