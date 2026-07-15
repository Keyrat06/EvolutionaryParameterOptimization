#!/usr/bin/env python3
"""Compare recursive and iterative Fibonacci implementations."""

import argparse
import json
import platform
import statistics
import timeit
from typing import Callable, Sequence


def fibonacci_recursive(n: int) -> int:
    """Return the nth Fibonacci number using recursive calls."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Return the nth Fibonacci number using a loop."""
    if n < 0:
        raise ValueError("n must be non-negative")

    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous


def benchmark(
    function: Callable[[int], int], n: int, repeat: int, number: int
) -> list[float]:
    """Return repeated timings, measured in seconds per function call."""
    timer = timeit.Timer(lambda: function(n))
    return [elapsed / number for elapsed in timer.repeat(repeat=repeat, number=number)]


def positive_integer(value: str) -> int:
    """Parse a positive integer for an argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative integer for an argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recursive and iterative Fibonacci execution speed."
    )
    parser.add_argument(
        "n",
        nargs="?",
        default=30,
        type=non_negative_integer,
        help="Fibonacci index to calculate (default: 30)",
    )
    parser.add_argument(
        "--repeat",
        default=5,
        type=positive_integer,
        help="number of timing samples (default: 5)",
    )
    parser.add_argument(
        "--number",
        type=positive_integer,
        help="function calls per sample for both styles (overrides style defaults)",
    )
    parser.add_argument(
        "--recursive-number",
        default=1,
        type=positive_integer,
        help="recursive calls per timing sample (default: 1)",
    )
    parser.add_argument(
        "--iterative-number",
        default=10000,
        type=positive_integer,
        help="iterative calls per timing sample (default: 10000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable benchmark results",
    )
    return parser.parse_args(argv)


def format_timings(name: str, timings: list[float]) -> str:
    best = min(timings)
    median = statistics.median(timings)
    return f"{name:<10} best: {best:.9f} s/call  median: {median:.9f} s/call"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recursive_number = args.number or args.recursive_number
    iterative_number = args.number or args.iterative_number

    recursive_result = fibonacci_recursive(args.n)
    iterative_result = fibonacci_iterative(args.n)
    if recursive_result != iterative_result:
        raise RuntimeError("Fibonacci implementations returned different results")

    recursive_timings = benchmark(
        fibonacci_recursive, args.n, args.repeat, recursive_number
    )
    iterative_timings = benchmark(
        fibonacci_iterative, args.n, args.repeat, iterative_number
    )

    recursive_best = min(recursive_timings)
    iterative_best = min(iterative_timings)
    speedup = recursive_best / iterative_best

    result = {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "n": args.n,
        "value": recursive_result,
        "repeat": args.repeat,
        "recursive": {
            "number": recursive_number,
            "best_seconds": recursive_best,
            "median_seconds": statistics.median(recursive_timings),
        },
        "iterative": {
            "number": iterative_number,
            "best_seconds": iterative_best,
            "median_seconds": statistics.median(iterative_timings),
        },
        "iterative_speedup": speedup,
    }

    if args.json:
        print(json.dumps(result))
        return 0

    print(f"Python:    {platform.python_version()} ({platform.python_implementation()})")
    print(f"F({args.n}):      {recursive_result}")
    print(
        f"Samples:   {args.repeat} x {recursive_number} recursive call(s), "
        f"{args.repeat} x {iterative_number} iterative call(s)"
    )
    print(format_timings("Recursive", recursive_timings))
    print(format_timings("Iterative", iterative_timings))
    print(f"Speedup:   iterative is {speedup:,.2f}x faster (best times)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
