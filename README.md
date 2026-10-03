# CAT Trust-Region Method — Python Implementation

An independent Python implementation of **CAT (Consistently Adaptive Trust Region)** from Fadi Hamad and Oliver Hinder's paper, [*A Simple and Practical Adaptive Trust-Region Method*](https://arxiv.org/abs/2412.02079).

This repository focuses on translating the method into a clear, modular research codebase: the CAT outer iteration, the inexact trust-region subproblem solver, shifted linear-system routines, CUTEst integration, experiment configuration, benchmark collection, and tests are separated into dedicated modules.

> **Attribution.** This is an independent implementation/reproduction of the method described by Hamad and Hinder. It is **not** the authors' official implementation. Their reference Julia repository is available at [fadihamad94/CATrustRegionMethod.jl](https://github.com/fadihamad94/CATrustRegionMethod.jl).

## Paper

Hamad and Hinder study unconstrained, possibly nonconvex optimization

```math
\min_{x \in \mathbb{R}^n} f(x),
```

using an adaptive trust-region method that permits inexact solutions of the trust-region subproblem. Under a Lipschitz-Hessian assumption, the paper establishes a first-order iteration complexity of

```math
O\!\left(\Delta_f L^{1/2}\epsilon^{-3/2}\right) + \widetilde{O}(1)
```

for finding an epsilon-approximate stationary point. The paper also evaluates the method on CUTEst and compares it with established trust-region and cubic-regularization methods.

**Paper:** [arXiv:2412.02079](https://arxiv.org/abs/2412.02079)  
**Official implementation:** [CATrustRegionMethod.jl](https://github.com/fadihamad94/CATrustRegionMethod.jl)

## What is implemented

- The CAT outer trust-region method, including adaptive radius updates, trial-point acceptance, gradient-based termination, and evaluation accounting.
- An inexact trust-region subproblem solver with Newton-step detection, interval search, bisection, and hard-case handling through inverse power iteration.
- Shifted systems of the form $(H + \delta I)\,x = b$, with sparse direct solves and a matrix-free iterative fallback.
- Sparse Hessian support through SciPy, optional CHOLMOD factorization through `scikit-sparse`, SuperLU fallback, and MINRES for `LinearOperator` Hessians.
- A small problem interface that decouples the optimizer from CUTEst and makes the solver usable with custom unconstrained problems.
- PyCUTEst integration for benchmark problems, resumable experiment execution, per-problem JSON output, and aggregate CSV results.
- YAML configuration for numerical parameters and benchmark settings.
- Unit tests for the CAT optimizer, trust-region subproblem solver, shifted-system solver, and CUTEst adapter.

## Repository structure

```text
Project/
├── problems/
│   ├── base.py                 # Minimal unconstrained-problem interface
│   ├── cutest.py               # PyCUTEst adapter
│   └── problem_names.txt       # CUTEst benchmark problem list
│
├── optimizers/
│   ├── cat.py                  # CAT outer algorithm
│   ├── trust_region.py         # Inexact trust-region subproblem solver
│   ├── linear_solver.py        # Shifted systems and PD checks
│   └── types.py                # Parameters, results, evaluation counters
│
├── benchmarks/
│   ├── cutest.py               # CUTEst benchmark runner
│   └── analysis.py             # Aggregate benchmark statistics
│
├── scripts/
│   ├── main.py                 # Main entry point
│   ├── run_benchmark.py        # Run benchmark experiments
│   └── analyze_results.py      # Summarize stored results
│
├── config/
│   ├── cat.yaml                # CAT and subproblem parameters
│   └── benchmark.yaml          # Benchmark configuration
│
├── results/
│   ├── raw/                    # Per-problem JSON results
│   ├── processed/              # Aggregate CSV results
│   └── figures/                # Reserved for generated figures
│
├── utils/
│   ├── config.py               # YAML/configuration helpers
│   └── linalg.py               # Numerical linear-algebra utilities
│
├── tests/                      # Unit tests
├── requirements.txt
├── setup.sh
└── README.md
```

## Algorithm-to-code map

| Mathematical component | Implementation |
| --- | --- |
| CAT outer iteration | `optimizers/cat.py` |
| Trust-region subproblem | `optimizers/trust_region.py` |
| $(H + \delta I)\,x = b$ solves / factorization | `optimizers/linear_solver.py` |
| Numerical parameters and result types | `optimizers/types.py` |
| Problem abstraction | `problems/base.py` |
| CUTEst adapter | `problems/cutest.py` |
| CUTEst experiment loop | `benchmarks/cutest.py` |
| Result aggregation | `benchmarks/analysis.py` |

The separation is intentional: the optimizer depends only on the small `UnconstrainedProblem` interface and is therefore not tied to CUTEst.

## Requirements

The full CUTEst benchmark workflow is intended for **Linux or WSL**. PyCUTEst itself supports Linux and macOS, but the included `setup.sh` is written for a Linux-style environment.

For Ubuntu/WSL, install the system dependencies first:

```bash
sudo apt update
sudo apt install -y \
    python3 python3-dev python3-venv \
    gcc gfortran meson git \
    libsuitesparse-dev
```

`libsuitesparse-dev` is required to build the optional `scikit-sparse`/CHOLMOD backend included in `requirements.txt`.

## Installation

After cloning the repository, run the setup script from the project root:

```bash
cd <repository-directory>
bash setup.sh
```

The setup script:

1. clones SIFDecode, CUTEst, and MASTSIF;
2. compiles and installs SIFDecode and CUTEst;
3. creates a Python virtual environment at `.env`;
4. installs the pinned Python dependencies from `requirements.txt`; and
5. configures `MASTSIF` and `PYCUTEST_CACHE` when the environment is activated.

Activate the environment with:

```bash
source .env/bin/activate
```

> The setup script installs SIFDecode and CUTEst system-wide via `sudo meson install`, so it may request your password.

## Running the CUTEst benchmark

From the repository root:

```bash
python -m scripts.run_benchmark
```

or equivalently:

```bash
python -m scripts.main
```

The benchmark runner reads `config/cat.yaml` and `config/benchmark.yaml`, loads the problem names from `problems/problem_names.txt`, and executes CAT on eligible CUTEst problems.

By default, completed problems already present in the aggregate results file are skipped, allowing interrupted benchmark runs to resume.

## Analyzing results

To print mean and median statistics from the aggregate CSV:

```bash
python -m scripts.analyze_results
```

The main output locations are:

```text
results/raw/<PROBLEM>.json
results/processed/cutest_results.csv
```

The raw JSON files include the final stationarity measure, iteration count, and internal evaluation counters. The CSV stores the benchmark-level quantities used by the experiment runner, including execution time and function, gradient, Hessian, and factorization counts.

## Configuration

Numerical settings are stored in `config/cat.yaml` rather than being hard-coded into experiment scripts. The current configuration includes the CAT parameters \(\beta,\theta,\omega_1,\omega_2,\gamma_1,\gamma_2,\gamma_3\), the stationarity tolerance, iteration limits, shifted linear-solver tolerances, and eigenvalue-estimation controls.

Benchmark selection is controlled by `config/benchmark.yaml`, including the problem list, maximum dimension, result paths, resume behavior, and problems that should be skipped in the current PyCUTEst setup.

This separation makes parameter studies and ablation experiments possible without modifying the optimization implementation itself.

## Using CAT on a custom problem

The solver is not restricted to CUTEst. A problem only needs to expose an initial point, objective, gradient, and Hessian compatible with `problems.base.UnconstrainedProblem`.

```python
import numpy as np
import scipy.sparse as sp

from optimizers import CATOptimizer, CATParams


class QuadraticProblem:
    def x0(self):
        return np.array([1.0, -2.0, 0.5])

    def obj(self, x):
        return 0.5 * float(x @ x)

    def grad(self, x):
        return np.asarray(x, dtype=float)

    def hess(self, x):
        return sp.eye(x.size, format="csc")


optimizer = CATOptimizer(
    CATParams(tol=1e-8, max_iter=100),
    record_history=True,
)
result = optimizer.minimize(QuadraticProblem())

print("status:", result.status)
print("f(x):", result.f)
print("stationarity:", result.eps)
print("iterations:", result.iters)
print("evaluations:", result.counters)
```

The Hessian may be supplied as a SciPy sparse matrix or as a `scipy.sparse.linalg.LinearOperator`. The latter enables matrix-free shifted solves through MINRES.

## Tests

Run the test suite from the project root:

```bash
pytest -q
```

The current tests cover convergence on a simple quadratic problem, trust-region Newton-step behavior, shifted-system accuracy, and the PyCUTEst adapter.

## Reproducibility notes

This repository is a Python reimplementation of the algorithmic ideas in the paper, whereas the authors' current reference implementation is written in Julia. Numerical details, linear-algebra backends, CUTEst versions, hardware, and stopping behavior can therefore affect exact iteration counts and timings.

The result files in this repository should be interpreted as outputs of **this implementation and environment**, not as a claim that they exactly reproduce every number reported in the paper. In particular, this repository currently focuses on CAT itself and does not implement the paper's complete external comparison pipeline for GALAHAD TRU/ARC or every ablation available in the authors' reference code.

## Research context

This implementation was developed while studying modern second-order methods for nonconvex optimization. The main engineering goal was to turn the mathematical method into a readable experimental system in which the outer algorithm, trust-region subproblem, numerical linear algebra, problem interface, and benchmarking infrastructure can be inspected and tested independently.

The repository began as a final project for **Advanced Optimization (Fall 2025), University of Tehran**, and was subsequently reorganized into the current research-oriented code structure.

## References and acknowledgements

The algorithm implemented here is based on:

> Fadi Hamad and Oliver Hinder. *A Simple and Practical Adaptive Trust-Region Method*. arXiv:2412.02079.

If you use the algorithm in research, please cite the original paper rather than this repository as the source of the method.

This project also relies on the [CUTEst](https://github.com/ralna/CUTEst) optimization test environment and [PyCUTEst](https://github.com/jfowkes/pycutest) for its Python interface to CUTEst.

For PyCUTEst, see:

> Jaroslav Fowkes, Lindon Roberts, and Árpád Bűrmen. “PyCUTEst: an open source Python package of optimization test problems.” *Journal of Open Source Software*, 7(78), 4377, 2022. https://doi.org/10.21105/joss.04377
