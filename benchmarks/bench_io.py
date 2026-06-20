#! /usr/bin/env python3
"""
benchmarks/bench_io.py — pandas vs Polars/Parquet I/O for mosna
===============================================================

Measures the speed of the data-loading layer that mosna uses everywhere (the
``read_fct = pd.read_parquet / pd.read_csv`` pattern, now routed through
:mod:`mosna.io`) and **verifies the results are identical** to pandas, so the
scientific logic is provably unchanged — only faster.

Run::

    python benchmarks/bench_io.py
    python benchmarks/bench_io.py --rows 500000 --cols 60 --repeats 5

It synthesises a wide numeric "nodes" table (the shape mosna handles: many
cells x many markers/coordinates), writes it as CSV and Parquet, then times:

  * read_parquet  : pandas  vs  mosna.io (Polars)
  * read_csv      : pandas  vs  mosna.io (Polars)
  * write_parquet : pandas  vs  mosna.io (Polars)
  * csv -> parquet conversion : pandas vs Polars

A correctness check asserts the Polars-loaded frame equals the pandas-loaded
frame (values + columns), aborting if any difference is found.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import mosna.io directly from its file. We avoid `from mosna import io`
# because the package __init__ pulls in the full (heavy) scientific stack,
# whereas the I/O layer only needs pandas/polars and should stand alone.
import importlib.util

_IO_PATH = Path(__file__).resolve().parent.parent / "mosna" / "io.py"
_spec = importlib.util.spec_from_file_location("mosna_io_standalone", _IO_PATH)
io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io)  # noqa: E402


def make_nodes(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic wide numeric table resembling mosna nodes (coords + markers)."""
    rng = np.random.default_rng(seed)
    data = {"x": rng.random(rows), "y": rng.random(rows)}
    for i in range(cols):
        data[f"marker_{i}"] = rng.random(rows)
    # a couple of categorical-ish columns like real node files have
    data["sample"] = rng.integers(0, 20, rows).astype("int64")
    return pd.DataFrame(data)


def timeit(fn, repeats: int) -> float:
    """Return the best (min) wall-clock time over `repeats` runs, in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def fmt(name: str, pd_t: float, pl_t: float) -> str:
    speedup = pd_t / pl_t if pl_t > 0 else float("inf")
    return (
        f"{name:<28} pandas={pd_t*1e3:8.1f} ms   "
        f"mosna.io={pl_t*1e3:8.1f} ms   speedup={speedup:5.2f}x"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--cols", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print(f"Polars available: {io.have_polars()}")
    if not io.have_polars():
        print("WARNING: polars not installed; mosna.io falls back to pandas, "
              "so no speedup is expected. Install with: pip install polars")
    print(f"Synthesising nodes table: {args.rows:,} rows x {args.cols+3} cols\n")

    df = make_nodes(args.rows, args.cols)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pq = d / "nodes.parquet"
        csv = d / "nodes.csv"
        df.to_parquet(pq, index=False)
        df.to_csv(csv, index=False)

        pq_mb = pq.stat().st_size / 1e6
        csv_mb = csv.stat().st_size / 1e6
        print(f"On disk: parquet={pq_mb:.1f} MB   csv={csv_mb:.1f} MB   "
              f"(parquet is {csv_mb/pq_mb:.1f}x smaller)\n")

        # ---- correctness: results must be identical to pandas ----
        ref = pd.read_parquet(pq)
        got = io.read_parquet(pq)
        pd.testing.assert_frame_equal(
            ref.reset_index(drop=True), got.reset_index(drop=True),
            check_dtype=False,  # polars may widen e.g. int dtypes; values equal
        )
        ref_csv = pd.read_csv(csv)
        got_csv = io.read_csv(csv)
        pd.testing.assert_frame_equal(
            ref_csv.reset_index(drop=True), got_csv.reset_index(drop=True),
            check_dtype=False,
        )
        print("Correctness check PASSED: mosna.io output == pandas output\n")

        # ---- benchmarks ----
        r = args.repeats
        results = []
        results.append(fmt(
            "read_parquet",
            timeit(lambda: pd.read_parquet(pq), r),
            timeit(lambda: io.read_parquet(pq), r),
        ))
        results.append(fmt(
            "read_csv",
            timeit(lambda: pd.read_csv(csv), r),
            timeit(lambda: io.read_csv(csv), r),
        ))
        out_pd = d / "out_pd.parquet"
        out_io = d / "out_io.parquet"
        results.append(fmt(
            "write_parquet",
            timeit(lambda: df.to_parquet(out_pd, index=False), r),
            timeit(lambda: io.write_parquet(df, out_io), r),
        ))
        out_conv = d / "conv.parquet"
        results.append(fmt(
            "csv -> parquet",
            timeit(lambda: pd.read_csv(csv).to_parquet(out_conv, index=False), r),
            timeit(lambda: io.csv_to_parquet(csv, out_conv), r),
        ))

        print("Results (best of {} runs):".format(r))
        print("-" * 78)
        for line in results:
            print(line)
        print("-" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
