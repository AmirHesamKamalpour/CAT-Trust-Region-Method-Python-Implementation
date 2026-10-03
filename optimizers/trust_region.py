from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from optimizers.linear_solver import ShiftedSystemSolver
from optimizers.types import Array, CATParams, EvalCounters
from problems.base import Hessian
from utils.linalg import (
    as_linear_operator,
    check_trs_conditions,
    matvec,
    norm2,
    quadratic_model,
)


class TrustRegionSubproblemSolver:
    """Appendix A trust-region subproblem solver (Algorithm 2)."""

    def __init__(self, params: CATParams, counters: EvalCounters):
        self.params = params
        self.counters = counters

    def solve(
        self,
        g: Array,
        H: Hessian,
        r: float,
        eps_k: float,
        delta_prev: float,
    ) -> Tuple[Array, float, Dict[str, Any]]:
        n = g.size
        H_mat = H if isinstance(H, sp.spmatrix) else None
        H_op = as_linear_operator(H, n)
        system = ShiftedSystemSolver(
            n=n,
            counters=self.counters,
            params=self.params,
            H_mat=H_mat,
            H_op=H_op,
        )

        stats: Dict[str, Any] = {"mode": None}
        can_try_newton = True
        if not system._cholmod_available:
            can_try_newton = system.is_pos_def(0.0)

        if can_try_newton:
            try:
                d_newton = system.solve(-g, 0.0)
                if norm2(d_newton) <= r:
                    ok, check = check_trs_conditions(
                        g, H, d_newton, 0.0, r, eps_k, self.params
                    )
                    if ok:
                        stats["mode"] = "newton"
                        stats["check"] = check
                        return d_newton, 0.0, stats
            except Exception:
                pass

        stats["mode"] = "alg2"
        d, delta, alg2_stats = self._algorithm2(
            g, H, system, r, eps_k, delta_prev
        )
        stats.update(alg2_stats)
        return d, delta, stats

    def _phi(
        self,
        g: Array,
        H: Hessian,
        system: ShiftedSystemSolver,
        r: float,
        eps_k: float,
        delta: float,
        cache: Dict[float, Tuple[Array, float, float, float]],
        counts: Optional[Dict[str, int]] = None,
    ) -> int:
        if counts is not None:
            counts["phi_evals"] = counts.get("phi_evals", 0) + 1

        if delta < 0:
            return +1

        if delta in cache:
            d, nd, stat, grad_model_norm = cache[delta]
        else:
            if not system._cholmod_available and not system.is_pos_def(delta):
                return +1
            try:
                d = np.asarray(system.solve(-g, delta)).reshape(-1)
            except Exception:
                return +1

            nd = norm2(d)
            Hd = matvec(H, d)
            grad = Hd + g
            grad_model_norm = norm2(grad)
            stat = norm2(grad + delta * d)
            cache[delta] = (d, nd, stat, grad_model_norm)

        eps = 1e-12 * max(1.0, r, eps_k)
        if nd > r + eps:
            return +1
        if nd < self.params.gamma2 * r - eps:
            return -1
        if (
            self.params.gamma2 * r - eps <= nd <= r + eps
            and stat <= self.params.gamma1 * eps_k + eps
        ):
            return 0
        if (
            delta == 0.0
            and nd <= r + eps
            and grad_model_norm <= self.params.gamma1 * eps_k + eps
        ):
            return 0
        return +1

    def _find_initial_interval(
        self,
        g: Array,
        H: Hessian,
        system: ShiftedSystemSolver,
        r: float,
        eps_k: float,
        delta_prev: float,
        cache: Dict[float, Tuple[Array, float, float, float]],
        counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[float, float]:
        d_in = float(delta_prev)
        phi_in = self._phi(g, H, system, r, eps_k, d_in, cache, counts)
        if phi_in == 0:
            return d_in, d_in

        d0 = 1.0 if d_in == 0.0 else d_in
        if not system._cholmod_available:
            d0 = max(d0, system.estimate_pd_delta_min())

        phi0 = self._phi(g, H, system, r, eps_k, d0, cache, counts)
        if phi0 == 0:
            return d0, d0
        phi_prev = int(phi0)

        log2_d0 = math.log2(d0)
        log2_min = math.log2(np.finfo(float).tiny)
        log2_max = math.log2(np.finfo(float).max)

        for i in range(1, self.params.trs_max_iter + 1):
            if i == 1:
                x = d0
            else:
                log2x = log2_d0 + ((i - 1) ** 2) * phi_prev
                log2x = min(max(log2x, log2_min), log2_max)
                x = 2.0**log2x

            log2y = log2_d0 + (i**2) * phi_prev
            log2y = min(max(log2y, log2_min), log2_max)
            y = 2.0**log2y

            phix = self._phi(g, H, system, r, eps_k, x, cache, counts)
            if phix == 0:
                return float(x), float(x)

            phiy = self._phi(g, H, system, r, eps_k, y, cache, counts)
            if phiy == 0:
                return float(y), float(y)

            if phix * phiy < 0:
                return float(min(x, y)), float(max(x, y))

            if log2y in (log2_min, log2_max):
                break

        raise RuntimeError("FIND-INITIAL-INTERVAL failed to bracket a root of phi.")

    def _bisection(
        self,
        g: Array,
        H: Hessian,
        system: ShiftedSystemSolver,
        r: float,
        eps_k: float,
        delta_lo: float,
        delta_hi: float,
        cache: Dict[float, Tuple[Array, float, float, float]],
        counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[float, float, float, Array, Array, bool]:
        eps = 1e-12 * max(1.0, r, eps_k)
        _ = self._phi(g, H, system, r, eps_k, delta_hi, cache, counts)

        # Degenerate interval means phi(delta)==0 was already found.
        if delta_lo == delta_hi:
            d = cache[delta_hi][0]
            return delta_lo, delta_hi, delta_hi, d, d, False

        for _ in range(self.params.trs_max_iter):
            if counts is not None:
                counts["bisect_iters"] = counts.get("bisect_iters", 0) + 1

            delta_mid = 0.5 * (delta_lo + delta_hi)
            phi_mid = self._phi(g, H, system, r, eps_k, delta_mid, cache, counts)

            if phi_mid == 0:
                d_mid = cache[delta_mid][0]
                d_hi = cache[delta_hi][0]
                return delta_lo, delta_mid, delta_hi, d_mid, d_hi, False

            if phi_mid == +1:
                delta_lo = delta_mid
            else:
                delta_hi = delta_mid

            width_ok = (
                True
                if r <= 0
                else (delta_hi - delta_lo)
                <= (self.params.gamma1 * eps_k) / (6.0 * r) + eps
            )
            d_hi, _, stat_hi, _ = cache[delta_hi]
            stat_ok = stat_hi <= (self.params.gamma1 * eps_k) / 3.0 + eps

            if width_ok and stat_ok:
                return delta_lo, delta_hi, delta_hi, d_hi, d_hi, True

        raise RuntimeError("BISECTION exceeded max iterations.")

    def _inverse_power_iteration(
        self,
        g: Array,
        H: Hessian,
        system: ShiftedSystemSolver,
        r: float,
        eps_k: float,
        delta_hi: float,
        d_hi: Array,
    ) -> Tuple[Array, float, Dict[str, Any]]:
        del d_hi  # retained in the signature to match Algorithm 2 bookkeeping
        n = g.size
        stats: Dict[str, Any] = {"hard_case": True, "ipi_iters": 0}

        for attempt in range(2):
            if attempt == 0:
                g_use = g
            else:
                u = np.random.randn(n)
                u /= max(norm2(u), 1e-300)
                g_use = g + 0.5 * self.params.gamma1 * eps_k * u
                stats["used_perturbed_gradient"] = True

            try:
                d0 = system.solve(-g_use, delta_hi)
            except Exception as exc:
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "Inverse power iteration: failed to solve shifted system"
                ) from exc

            y = np.random.randn(n)
            y /= max(norm2(y), 1e-300)

            for it in range(self.params.trs_max_iter):
                stats["ipi_iters"] = it + 1
                y = np.asarray(system.solve(y, delta_hi)).reshape(-1)
                y /= max(norm2(y), 1e-300)

                b = float(y @ d0)
                c = float(d0 @ d0 - r * r)
                disc = max(0.0, b * b - c)
                sqrt_disc = math.sqrt(disc)

                candidates = []
                for alpha in (-b + sqrt_disc, -b - sqrt_disc):
                    d = d0 + alpha * y
                    ok, check = check_trs_conditions(
                        g_use, H, d, delta_hi, r, eps_k, self.params
                    )
                    if ok:
                        candidates.append((d, check))

                if candidates:
                    best_d, best_check = min(
                        candidates,
                        key=lambda item: quadratic_model(g_use, H, item[0]),
                    )
                    stats["check"] = best_check
                    return best_d, float(delta_hi), stats

        raise RuntimeError(
            "Inverse power iteration failed to produce a direction satisfying (6)."
        )

    def _algorithm2(
        self,
        g: Array,
        H: Hessian,
        system: ShiftedSystemSolver,
        r: float,
        eps_k: float,
        delta_prev: float,
    ) -> Tuple[Array, float, Dict[str, Any]]:
        cache: Dict[float, Tuple[Array, float, float, float]] = {}
        stats: Dict[str, Any] = {"hard_case": False}
        counts: Dict[str, int] = {}

        delta_lo, delta_hi = self._find_initial_interval(
            g, H, system, r, eps_k, delta_prev, cache, counts
        )
        dlo, dmid, dhi, d_mid, d_hi, hard = self._bisection(
            g, H, system, r, eps_k, delta_lo, delta_hi, cache, counts
        )

        stats.update(
            {
                "delta_lo": float(dlo),
                "delta_mid": float(dmid),
                "delta_hi": float(dhi),
                **counts,
            }
        )

        if not hard:
            delta = float(dmid)
            d = np.asarray(d_mid).reshape(-1)
            ok, check = check_trs_conditions(
                g, H, d, delta, r, eps_k, self.params
            )
            stats["check"] = check
            if not ok:
                raise RuntimeError(
                    "Bisection returned phi=0 but condition (6) failed numerically."
                )
            return d, delta, stats

        stats["hard_case"] = True
        d, delta, ipi_stats = self._inverse_power_iteration(
            g, H, system, r, eps_k, float(dhi), np.asarray(d_hi).reshape(-1)
        )
        stats.update(ipi_stats)
        return d, delta, stats
