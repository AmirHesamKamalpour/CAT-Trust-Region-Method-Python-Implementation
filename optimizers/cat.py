from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from optimizers.trust_region import TrustRegionSubproblemSolver
from optimizers.types import Array, CATParams, CATResult, EvalCounters
from problems.base import UnconstrainedProblem
from utils.linalg import estimate_spectral_norm, norm2, quadratic_model


class CATOptimizer:
    """Consistently Adaptive Trust-Region method (Algorithm 1)."""

    def __init__(
        self,
        params: Optional[CATParams] = None,
        record_history: bool = False,
    ):
        self.params = params or CATParams()
        self.record_history = record_history

    def minimize(self, prob: UnconstrainedProblem) -> CATResult:
        p = self.params
        counters = EvalCounters()
        trs = TrustRegionSubproblemSolver(params=p, counters=counters)
        t0 = time.time()

        x = np.asarray(prob.x0(), dtype=float).reshape(-1)
        f = float(prob.obj(x))
        counters.f += 1
        g = np.asarray(prob.grad(x), dtype=float).reshape(-1)
        counters.g += 1
        H = prob.hess(x)
        counters.H += 1

        eps_k = norm2(g)
        x_eps = x.copy()
        f_eps = float(f)

        Hnorm = estimate_spectral_norm(H, x.size, counters, p)
        r = 1.0 if Hnorm == 0.0 else 10.0 * eps_k / Hnorm
        delta_prev = 0.0

        history: Optional[List[Dict[str, Any]]] = [] if self.record_history else None
        x_trial_buf = np.empty_like(x)

        for k in range(1, p.max_iter + 1):
            now = time.time()
            if p.max_time_sec is not None and (now - t0) > p.max_time_sec:
                return CATResult(
                    x=x,
                    f=f,
                    eps=eps_k,
                    status="MAX_TIME",
                    iters=k - 1,
                    counters=counters,
                    runtime_sec=now - t0,
                    history=history,
                )

            if eps_k <= p.tol:
                return CATResult(
                    x=x_eps,
                    f=f_eps,
                    eps=eps_k,
                    status="SUCCESS",
                    iters=k - 1,
                    counters=counters,
                    runtime_sec=now - t0,
                    history=history,
                )

            d, delta, trs_stats = trs.solve(
                g=g,
                H=H,
                r=r,
                eps_k=eps_k,
                delta_prev=delta_prev,
            )
            nd = norm2(d)

            if nd < p.step_min:
                return CATResult(
                    x=x,
                    f=f,
                    eps=eps_k,
                    status="STEP_TOO_SMALL",
                    iters=k - 1,
                    counters=counters,
                    runtime_sec=now - t0,
                    history=history,
                )

            np.add(x, d, out=x_trial_buf)
            x_trial = x_trial_buf.copy()
            f_trial = float(prob.obj(x_trial))
            counters.f += 1

            b_k = 0.1 * eps_k * nd + 1e-8 * (abs(f) + 1.0)

            g_trial: Optional[Array] = None
            gtrial_norm: Optional[float] = None
            if f_trial <= f + b_k:
                g_trial = np.asarray(prob.grad(x_trial), dtype=float).reshape(-1)
                counters.g += 1
                gtrial_norm = norm2(g_trial)

                if gtrial_norm <= eps_k:
                    x_eps = x_trial.copy()
                    f_eps = float(f_trial)

                if gtrial_norm <= p.tol:
                    now2 = time.time()
                    return CATResult(
                        x=x_trial,
                        f=f_trial,
                        eps=gtrial_norm,
                        status="SUCCESS",
                        iters=k,
                        counters=counters,
                        runtime_sec=now2 - t0,
                        history=history,
                    )

            gk_norm = norm2(g)
            min_g = gk_norm if g_trial is None else min(gk_norm, gtrial_norm)  # type: ignore[arg-type]

            Mk = quadratic_model(g, H, d)
            predicted = -Mk + 0.5 * p.theta * min_g * nd
            rho_hat = -float("inf") if predicted <= 0.0 else (f - f_trial) / predicted

            accepted = f_trial <= f
            if accepted:
                x = x_trial
                f = f_trial
                if g_trial is None:
                    # Accepted implies f_trial <= f_old <= f_old + b_k, hence gradient was evaluated.
                    raise RuntimeError("Accepted CAT step is missing its trial gradient.")
                g = g_trial
                H = prob.hess(x)
                counters.H += 1

            eps_next = eps_k
            if g_trial is not None:
                assert gtrial_norm is not None
                eps_next = min(eps_next, gtrial_norm)

            successful = rho_hat >= p.beta
            r = max(p.omega2 * nd, r) if successful else r / p.omega1

            eps_k = float(eps_next)
            delta_prev = float(delta)

            if history is not None:
                history.append(
                    {
                        "iter": k,
                        "f": f,
                        "eps": eps_k,
                        "r": r,
                        "||d||": nd,
                        "delta": delta,
                        "rho_hat": rho_hat,
                        "predicted": predicted,
                        "predicted_le_zero": predicted <= 0,
                        "accepted": accepted,
                        "successful": successful,
                        "trs_mode": trs_stats.get("mode"),
                        "trs_hard": trs_stats.get("hard_case", False),
                        "phi_evals": trs_stats.get("phi_evals", 0),
                        "bisect_iters": trs_stats.get("bisect_iters", 0),
                        "ipi_iters": trs_stats.get("ipi_iters", 0),
                        "x_eps_f": f_eps,
                    }
                )

        now = time.time()
        return CATResult(
            x=x,
            f=f,
            eps=eps_k,
            status="MAX_ITER",
            iters=p.max_iter,
            counters=counters,
            runtime_sec=now - t0,
            history=history,
        )
