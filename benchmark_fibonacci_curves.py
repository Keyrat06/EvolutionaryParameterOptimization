#!/usr/bin/env python3
"""Plot log-log Fibonacci timing curves across Python versions."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INTERPRETERS = [f"python3.{minor}" for minor in range(10, 15)]
DEFAULT_N_VALUES = [5, 10, 15, 20, 25, 30, 35]
ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "fibonacci_benchmark.py"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recursive and iterative Fibonacci curves by Python version."
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="interpreters",
        metavar="EXECUTABLE",
        help="Python executable to benchmark; repeat for multiple versions",
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        default=DEFAULT_N_VALUES,
        type=positive_integer,
        help="positive Fibonacci indexes (default: 5 10 15 20 25 30 35)",
    )
    parser.add_argument(
        "--repeat",
        default=3,
        type=positive_integer,
        help="timing samples per point (default: 3)",
    )
    parser.add_argument(
        "--recursive-number",
        type=positive_integer,
        help="fixed recursive calls per sample (default: adapt to n)",
    )
    parser.add_argument(
        "--iterative-number",
        default=100000,
        type=positive_integer,
        help="iterative calls per sample (default: 100000)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmark_results/fibonacci_curves"),
        help="path prefix for .csv and .png output",
    )
    return parser.parse_args(argv)


def resolve_interpreters(names: list[str]) -> tuple[list[str], list[str]]:
    resolved = []
    missing = []
    for name in names:
        executable = shutil.which(name)
        if executable:
            resolved.append(executable)
        else:
            missing.append(name)
    return resolved, missing


def adaptive_recursive_number(n: int) -> int:
    """Use enough calls for stable small-n timings without slowing large n."""
    return 10 ** max(0, (30 - n) // 5)


def run_point(
    executable: str, n: int, args: argparse.Namespace
) -> dict[str, Any]:
    recursive_number = args.recursive_number or adaptive_recursive_number(n)
    command = [
        executable,
        str(WORKER),
        str(n),
        "--repeat",
        str(args.repeat),
        "--recursive-number",
        str(recursive_number),
        "--iterative-number",
        str(args.iterative_number),
        "--json",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    result = json.loads(completed.stdout)
    result["executable"] = Path(executable).name
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "python_version",
        "implementation",
        "n",
        "value",
        "recursive_number",
        "recursive_median_ms",
        "iterative_number",
        "iterative_median_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "python_version": result["python_version"],
                    "implementation": result["implementation"],
                    "n": result["n"],
                    "value": result["value"],
                    "recursive_number": result["recursive"]["number"],
                    "recursive_median_ms": (
                        result["recursive"]["median_seconds"] * 1000
                    ),
                    "iterative_number": result["iterative"]["number"],
                    "iterative_median_ms": (
                        result["iterative"]["median_seconds"] * 1000
                    ),
                }
            )


def write_png(path: Path, results: list[dict[str, Any]]) -> None:
    versions = list(dict.fromkeys(result["python_version"] for result in results))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    for version in versions:
        version_results = [
            result for result in results if result["python_version"] == version
        ]
        version_results.sort(key=lambda result: result["n"])
        n_values = [result["n"] for result in version_results]

        for axis, style in zip(axes, ("recursive", "iterative")):
            milliseconds = [
                result[style]["median_seconds"] * 1000
                for result in version_results
            ]
            axis.plot(
                n_values,
                milliseconds,
                marker="o",
                linewidth=2,
                markersize=4,
                label=f"Python {version}",
            )

    for axis, title in zip(axes, ("Recursive Fibonacci", "Iterative Fibonacci")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("n (log scale)")
        axis.set_ylabel("Median time per call, ms (log scale)")
        axis.set_xticks(sorted(set(result["n"] for result in results)))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(versions))
    figure.suptitle("Fibonacci performance across CPython versions", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.interpreters or DEFAULT_INTERPRETERS
    interpreters, missing = resolve_interpreters(names)
    if missing:
        print(f"Skipping unavailable interpreters: {', '.join(missing)}", file=sys.stderr)
    if not interpreters:
        print("No requested Python interpreters were found.", file=sys.stderr)
        return 2

    n_values = sorted(set(args.n_values))
    results = []
    for executable in interpreters:
        for n in n_values:
            print(f"Benchmarking {Path(executable).name} at n={n}...", file=sys.stderr)
            results.append(run_point(executable, n, args))

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    write_csv(csv_path, results)
    write_png(png_path, results)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
