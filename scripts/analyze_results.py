from pathlib import Path

from benchmarks.analysis import print_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print_summary(PROJECT_ROOT / "results" / "processed" / "cutest_results.csv")


if __name__ == "__main__":
    main()
