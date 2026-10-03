from pathlib import Path

from benchmarks.cutest import run_cutest_benchmark
from utils.config import load_cat_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    params = load_cat_params(PROJECT_ROOT / "config" / "cat.yaml")
    run_cutest_benchmark(params=params)


if __name__ == "__main__":
    main()
