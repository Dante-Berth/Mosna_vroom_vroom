#! /usr/bin/env python3
"""
benchmarks/bench_compute.py — refactored vs original mosna, per optimised kernel
================================================================================

Benchmarks the three compute optimisations that the refactor introduced, each
against Alexis Coullomb's original implementation, and **verifies the results
match** before reporting timings:

  1. mixing_matrix            (bincount fast path for one-hot attributes)
  2. randomized_mixmat        (codes-only serial shuffle loop)
  3. aggregate_k_neighbors    (order-1 vectorised mean/std via sparse products)

The original is loaded from ``tests/_orig_mosna_reference.py``. If it is not
present, only the refactored timings are shown.

Run::

    python benchmarks/bench_compute.py
    python benchmarks/bench_compute.py --cells 500000 --neighbors 8 \
        --attributes 10 --n-shuffle 30 --nas-cells 50000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_original():
    ref = ROOT / "tests" / "_orig_mosna_reference.py"
    if not ref.exists():
        return None
    spec = importlib.util.spec_from_file_location("orig_mosna_bench", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_graph(n_cells, n_neighbors, n_markers, n_attributes, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((n_cells, 2)) * np.sqrt(n_cells)
    _, idx = cKDTree(coords).query(coords, k=n_neighbors + 1, workers=-1)
    src = np.repeat(np.arange(n_cells), n_neighbors)
    dst = idx[:, 1:].ravel()
    e = np.unique(np.sort(np.stack([src, dst], axis=1), axis=1), axis=0)
    edges = pd.DataFrame(e, columns=["source", "target"])

    markers = rng.random((n_cells, n_markers))
    onehot = np.zeros((n_cells, n_attributes), dtype=int)
    onehot[np.arange(n_cells), rng.integers(0, n_attributes, n_cells)] = 1
    attrs = [f"type_{k}" for k in range(n_attributes)]
    marker_cols = [f"marker_{i}" for i in range(n_markers)]
    nodes = pd.concat([
        pd.DataFrame(coords, columns=["x", "y"]),
        pd.DataFrame(markers, columns=marker_cols),
        pd.DataFrame(onehot, columns=attrs),
    ], axis=1)
    return nodes, edges, attrs, marker_cols


def timeit(fn, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def safe_timeit(fn, repeats):
    """Like timeit but tolerant of the *original* baseline failing (e.g. the
    upstream code uses np.asmatrix + float(), which raises on NumPy >= 2.3).
    Returns (None, None) on error so the benchmark still reports our timing."""
    try:
        return timeit(fn, repeats)
    except Exception as exc:  # noqa: BLE001 - baseline is third-party, may break
        print(f"   [original baseline unavailable: {type(exc).__name__}: {exc}]")
        return None, None


def report(name, t_orig, t_new, identical):
    flag = "OK" if identical else "MISMATCH!"
    if t_orig is None:
        print(f"{name:<26} original=     n/a      new={t_new*1e3:9.1f} ms")
    else:
        sp = t_orig / t_new if t_new else float("inf")
        print(f"{name:<26} original={t_orig*1e3:9.1f} ms   "
              f"new={t_new*1e3:9.1f} ms   speedup={sp:7.0f}x   [{flag}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=100_000)
    ap.add_argument("--neighbors", type=int, default=8)
    ap.add_argument("--markers", type=int, default=20)
    ap.add_argument("--attributes", type=int, default=10)
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--nas-cells", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    import mosna as new
    orig = load_original()

    print("=" * 80)
    print("mosna compute benchmark — original (A. Coullomb) vs refactored")
    print("=" * 80)
    if orig is None:
        print("NOTE: original reference not found; showing refactored timings only.\n")

    print(f"Building graph: {args.cells:,} cells, kNN={args.neighbors}, "
          f"{args.attributes} cell types, {args.markers} markers ...")
    nodes, edges, attrs, marker_cols = build_graph(
        args.cells, args.neighbors, args.markers, args.attributes)
    print(f"  -> {len(edges):,} undirected edges\n")
    r = args.repeats

    # 1. mixing_matrix
    t_new, mm_new = timeit(lambda: new.mixing_matrix(nodes, edges, attrs), r)
    t_orig = None
    identical = True
    if orig is not None:
        t_orig, mm_orig = safe_timeit(lambda: orig.mixing_matrix(nodes, edges, attrs), r)
        if mm_orig is not None:
            identical = np.array_equal(mm_orig, mm_new)
    report("1. mixing_matrix", t_orig, t_new, identical)

    # 2. randomized_mixmat (serial)
    ns = args.n_shuffle
    t_new2, _ = timeit(lambda: new.randomized_mixmat(
        nodes, edges, attrs, n_shuffle=ns, parallel=False, verbose=0,
        random_state=0), 1)
    t_orig2 = None
    ident2 = True  # different RNG draws by design; checked statistically in tests
    if orig is not None:
        t_orig2, _ = safe_timeit(lambda: orig.randomized_mixmat(
            nodes, edges, attrs, n_shuffle=ns, parallel=False, verbose=0), 1)
    report(f"2. randomized_mixmat(x{ns})", t_orig2, t_new2, ident2)

    # 3. aggregate_k_neighbors (order 1)
    sub = min(args.nas_cells, args.cells)
    es = edges[(edges["source"] < sub) & (edges["target"] < sub)].values
    X = nodes[marker_cols].values[:sub]
    t_new3, agg_new = timeit(lambda: new.aggregate_k_neighbors(
        X, es, order=1, var_names=marker_cols), 1)
    t_orig3 = None
    ident3 = True
    if orig is not None:
        t_orig3, agg_orig = safe_timeit(lambda: orig.aggregate_k_neighbors(
            X, es, order=1, var_names=marker_cols), 1)
        if agg_orig is not None:
            ident3 = np.allclose(agg_orig.values, agg_new[agg_orig.columns].values,
                                 rtol=1e-8, atol=1e-8)
    report(f"3. aggregate_k_neighbors({sub:,})", t_orig3, t_new3, ident3)

    print("-" * 80)
    print("Note: randomized_mixmat draws differ from the original by design "
          "(codes-only\nshuffle); equivalence is asserted statistically in "
          "tests/test_equivalence.py.")


if __name__ == "__main__":
    raise SystemExit(main())
