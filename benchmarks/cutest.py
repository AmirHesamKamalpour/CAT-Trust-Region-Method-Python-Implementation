from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import pycutest
from tqdm.auto import tqdm

from benchmarks.analysis import print_summary
from optimizers import CATOptimizer, CATParams
from problems import PyCUTEstProblem
from utils.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = [
    "problem_name",
    "status",
    "total_execution_time",
    "function_value",
    "gradient_value",
    "total_function_evaluation",
    "total_gradient_evaluation",
    "total_hessian_evaluation",
    "total_factorization_evaluation",
]


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_problem_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _completed_names(results_csv: Path) -> Set[str]:
    if not results_csv.exists():
        return set()
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["problem_name"]
            for row in csv.DictReader(handle)
            if row.get("problem_name")
        }


def _ensure_results_csv(results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    if results_csv.exists() and results_csv.stat().st_size > 0:
        return
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()


def _write_raw_result(raw_dir: Path, name: str, row: Dict[str, Any], result: Any) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **row,
        "eps": float(result.eps),
        "iterations": int(result.iters),
        "counters": asdict(result.counters),
    }
    with (raw_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_cutest_benchmark(
    params: Optional[CATParams] = None,
    benchmark_config: Optional[Path] = None,
) -> None:
    config_path = benchmark_config or (PROJECT_ROOT / "config" / "benchmark.yaml")
    config = load_yaml(config_path)

    problem_names_path = _resolve_project_path(config["problem_names"])
    results_csv = _resolve_project_path(config["results_csv"])
    raw_results_dir = _resolve_project_path(config["raw_results_dir"])
    max_dimension = int(config.get("max_dimension", 10000))
    resume = bool(config.get("resume", True))
    quiet_import = bool(config.get("quiet_import", False))
    skip_names = set(config.get("skip_names", []))

    problem_names = _read_problem_names(problem_names_path)
    _ensure_results_csv(results_csv)
    completed = _completed_names(results_csv) if resume else set()
    optimizer = CATOptimizer(params=params)

    print("Loading CUTEst benchmark set...")
    if completed:
        print(f"Resuming with {len(completed)} existing result(s).")

    with results_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)

        for name in (progress := tqdm(problem_names)):
            progress.set_description(name)

            if name in skip_names or name in completed:
                continue

            properties = pycutest.problem_properties(name)
            n_property = properties.get("n")
            if n_property != "variable" and int(n_property) > max_dimension:
                continue

            problem = pycutest.import_problem(name, quiet=quiet_import)
            if problem.n > max_dimension:
                pycutest.CUTEstProblem._instances[name] = None
                continue

            try:
                result = optimizer.minimize(PyCUTEstProblem(problem))
                report = problem.report()
                row: Dict[str, Any] = {
                    "problem_name": name,
                    "status": result.status,
                    "total_execution_time": result.runtime_sec,
                    "function_value": result.f,
                    "gradient_value": 0,  # preserved from the original CSV schema
                    "total_function_evaluation": report["f"],
                    "total_gradient_evaluation": report["g"],
                    "total_hessian_evaluation": report["H"],
                    "total_factorization_evaluation": result.counters.fact,
                }
                writer.writerow(row)
                handle.flush()
                _write_raw_result(raw_results_dir, name, row, result)
            except RuntimeError as exc:
                print(f"{name}: {exc}")
            finally:
                # pycutest keeps a global instance registry. Clearing the entry mirrors
                # the memory-release workaround used by the original project.
                pycutest.CUTEstProblem._instances[name] = None

    print_summary(results_csv)
