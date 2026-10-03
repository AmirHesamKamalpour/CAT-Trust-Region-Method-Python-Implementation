from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np

PathLike = Union[str, Path]

NUMERIC_COLUMNS = (
    "total_execution_time",
    "function_value",
    "gradient_value",
    "total_function_evaluation",
    "total_gradient_evaluation",
    "total_hessian_evaluation",
    "total_factorization_evaluation",
)


def load_result_rows(results_csv: PathLike) -> List[Dict[str, str]]:
    path = Path(results_csv)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_results(results_csv: PathLike) -> Dict[str, Dict[str, float]]:
    rows = load_result_rows(results_csv)
    summary: Dict[str, Dict[str, float]] = {}
    for column in NUMERIC_COLUMNS:
        values = [float(row[column]) for row in rows if row.get(column, "") != ""]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary[column] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
        }
    return summary


def print_summary(results_csv: PathLike) -> None:
    summary = summarize_results(results_csv)
    if not summary:
        print("No benchmark results found.")
        return

    print(f"Results: {Path(results_csv)}")
    print(f"{'metric':38s} {'mean':>16s} {'median':>16s}")
    print("-" * 72)
    for metric, stats in summary.items():
        print(f"{metric:38s} {stats['mean']:16.6g} {stats['median']:16.6g}")
