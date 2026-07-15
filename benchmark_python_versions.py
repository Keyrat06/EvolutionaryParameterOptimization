#!/usr/bin/env python3
"""Run the Fibonacci benchmark with multiple Python interpreters and plot it."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape


DEFAULT_INTERPRETERS = [f"python3.{minor}" for minor in range(10, 15)]
ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "fibonacci_benchmark.py"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Fibonacci styles across Python versions."
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="interpreters",
        metavar="EXECUTABLE",
        help="Python executable to benchmark; repeat for multiple versions",
    )
    parser.add_argument(
        "--n",
        default=30,
        type=non_negative_integer,
        help="Fibonacci index (default: 30)",
    )
    parser.add_argument(
        "--repeat",
        default=5,
        type=positive_integer,
        help="timing samples per style and version (default: 5)",
    )
    parser.add_argument(
        "--recursive-number",
        default=1,
        type=positive_integer,
        help="recursive calls per sample (default: 1)",
    )
    parser.add_argument(
        "--iterative-number",
        default=10000,
        type=positive_integer,
        help="iterative calls per sample (default: 10000)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmark_results/fibonacci_python_versions"),
        help="path prefix for .csv and .svg output",
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


def run_benchmark(executable: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        executable,
        str(WORKER),
        str(args.n),
        "--repeat",
        str(args.repeat),
        "--recursive-number",
        str(args.recursive_number),
        "--iterative-number",
        str(args.iterative_number),
        "--json",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    result = json.loads(completed.stdout)
    result["executable"] = executable
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "python_version",
        "implementation",
        "executable",
        "n",
        "value",
        "recursive_best_seconds",
        "recursive_median_seconds",
        "iterative_best_seconds",
        "iterative_median_seconds",
        "iterative_speedup",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "python_version": result["python_version"],
                    "implementation": result["implementation"],
                    "executable": result["executable"],
                    "n": result["n"],
                    "value": result["value"],
                    "recursive_best_seconds": result["recursive"]["best_seconds"],
                    "recursive_median_seconds": result["recursive"]["median_seconds"],
                    "iterative_best_seconds": result["iterative"]["best_seconds"],
                    "iterative_median_seconds": result["iterative"]["median_seconds"],
                    "iterative_speedup": result["iterative_speedup"],
                }
            )


def write_svg(path: Path, results: list[dict[str, Any]]) -> None:
    width, height = 960, 560
    left, right, top, bottom = 100, 35, 65, 100
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = [
        result[style]["median_seconds"]
        for result in results
        for style in ("recursive", "iterative")
    ]
    log_min = math.floor(math.log10(min(values)))
    log_max = math.ceil(math.log10(max(values)))
    if log_min == log_max:
        log_max += 1

    def y_position(value: float) -> float:
        fraction = (math.log10(value) - log_min) / (log_max - log_min)
        return top + plot_height * (1 - fraction)

    colors = {"recursive": "#4c78a8", "iterative": "#f58518"}
    group_width = plot_width / len(results)
    bar_width = min(52, group_width * 0.3)
    baseline = top + plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Fibonacci speed across CPython versions</text>',
        f'<text x="480" y="51" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13">F({results[0]["n"]}), median seconds per call (log scale)</text>',
    ]

    for exponent in range(log_min, log_max + 1):
        y = y_position(10**exponent)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
                'stroke="#d9d9d9" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">1e{exponent} s</text>',
            ]
        )

    for index, result in enumerate(results):
        center = left + group_width * (index + 0.5)
        for style, offset in (("recursive", -bar_width / 2), ("iterative", bar_width / 2)):
            value = result[style]["median_seconds"]
            y = y_position(value)
            x = center + offset - bar_width / 2
            label = f'{style.title()}: {value:.3e} seconds'
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{baseline - y:.2f}" fill="{colors[style]}">'
                f"<title>{escape(label)}</title></rect>"
            )
        version = escape(result["python_version"])
        speedup = result["iterative_speedup"]
        lines.extend(
            [
                f'<text x="{center:.2f}" y="{baseline + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13">Python {version}</text>',
                f'<text x="{center:.2f}" y="{baseline + 44}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{speedup:,.0f}x speedup</text>',
            ]
        )

    legend_y = height - 25
    lines.extend(
        [
            f'<rect x="350" y="{legend_y - 12}" width="16" height="16" '
            f'fill="{colors["recursive"]}"/>',
            f'<text x="373" y="{legend_y + 1}" font-family="sans-serif" '
            'font-size="13">Recursive</text>',
            f'<rect x="485" y="{legend_y - 12}" width="16" height="16" '
            f'fill="{colors["iterative"]}"/>',
            f'<text x="508" y="{legend_y + 1}" font-family="sans-serif" '
            'font-size="13">Iterative</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.interpreters or DEFAULT_INTERPRETERS
    interpreters, missing = resolve_interpreters(names)
    if missing:
        print(f"Skipping unavailable interpreters: {', '.join(missing)}", file=sys.stderr)
    if not interpreters:
        print("No requested Python interpreters were found.", file=sys.stderr)
        return 2

    results = []
    for executable in interpreters:
        print(f"Benchmarking {executable}...", file=sys.stderr)
        results.append(run_benchmark(executable, args))

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    svg_path = output_prefix.with_suffix(".svg")
    write_csv(csv_path, results)
    write_svg(svg_path, results)

    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
