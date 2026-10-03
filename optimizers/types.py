from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

Array = np.ndarray


@dataclass
class EvalCounters:
    """Function, derivative, Hessian-vector, and factorization counters."""

    f: int = 0
    g: int = 0
    H: int = 0
    Hv: int = 0
    fact: int = 0


@dataclass
class CATParams:
    """Numerical parameters for the CAT outer method and TR subproblem solver."""

    # Algorithm 1 defaults
    beta: float = 0.1
    theta: float = 0.1
    omega1: float = 8.0
    omega2: float = 16.0

    gamma1: float = 0.01
    gamma2: float = 0.8
    gamma3: float = 0.5

    # Practical settings
    tol: float = 1e-5
    max_iter: int = 100000
    step_min: float = 2e-16
    max_time_sec: Optional[float] = None

    # Trust-region subproblem controls
    trs_max_iter: int = 100
    lin_solve_tol: float = 1e-10
    lin_solve_maxit: int = 5000

    # Eigenvalue-estimation controls
    eig_maxiter: int = 200
    eig_tol: float = 1e-6

    # Small regularization used by the fallback PD test
    pd_jitter: float = 1e-12


@dataclass
class CATResult:
    """Result returned by :class:`CATOptimizer`."""

    x: Array
    f: float
    eps: float
    status: str
    iters: int
    counters: EvalCounters
    runtime_sec: float
    history: Optional[List[Dict[str, Any]]] = None
