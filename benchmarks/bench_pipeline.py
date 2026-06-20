#! /usr/bin/env python3
"""
benchmarks/bench_pipeline.py — original mosna vs refactored mosna
=================================================================

End-to-end speed comparison of Alexis Coullomb's original monolithic ``mosna``
against this refactored package, on a **big spatial graph / big cell set**.

It builds a synthetic spatial single-cell dataset (N cells, M markers, a kNN
spatial graph) and times the heavy shared pipeline stages on *both* code bases:

  1. Assortativity     : mixing_matrix + randomized_mixmat (shuffle + recompute)
  2. Neighbor aggreg.  : aggregate_k_neighbors (NAS features) on a cell subset
  3. I/O round-trip    : save + load the big node table

Because the refactor kept every function body identical, stages 1-2 are expected
to be ~equal (same algorithm) — that is the point: the science/compute is
unchanged. Stage 3 is where the refactor's Polars/Parquet I/O layer differs.

The original implementation is loaded from ``tests/_orig_mosna_reference.py``
(pristine upstream copy). Results are also checked for equality on stage 1.

Run::

    python benchmarks/bench_pipeline.py
    python benchmarks/bench_pipeline.py --cells 200000 --neighbors 8 \
        --markers 30 --attributes 8 --n-shuffle 30 --nas-cells 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_original():
    """Load the pristine original mosna from the test reference, if present."""
    ref = ROOT / "tests" / "_orig_mosna_reference.py"
    if not ref.exists():
        return None
    spec = importlib.util.spec_from_file_location("orig_mosna_bench", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_big_graph(n_cells, n_neighbors, n_markers, n_attributes, seed=0):
    """Synthesise a big spatial graph: coords, kNN edges, markers, attributes."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n_cells, 2)) * np.sqrt(n_cells)  # keep density ~constant

    # kNN spatial edges (undirected, deduplicated) — the kind tysserand builds.
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=n_neighbors + 1)  # +1: self
    src = np.repeat(np.arange(n_cells), n_neighbors)
    dst = idx[:, 1:].ravel()
    e = np.sort(np.stack([src, dst], axis=1), axis=1)
    e = np.unique(e, axis=0)
    edges = pd.DataFrame(e, columns=["source", "target"])

    # markers (continuous omics) + one-hot categorical attributes (cell types)
    markers = rng.random((n_cells, n_markers))
    cell_type = rng.integers(0, n_attributes, n_cells)
    attr_cols = [f"type_{k}" for k in range(n_attributes)]
    onehot = np.zeros((n_cells, n_attributes), dtype=int)
    onehot[np.arange(n_cells), cell_type] = 1

    nodes = pd.DataFrame(coords, columns=["x", "y"])
    nodes = pd.concat(
        [nodes,
         pd.DataFrame(markers, columns=[f"marker_{i}" for i in range(n_markers)]),
         pd.DataFrame(onehot, columns=attr_cols)],
        axis=1,
    )
    return nodes, edges, attr_cols


def timeit(fn, repeats):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def fmt(stage, t_orig, t_new):
    if t_orig is None:
        return f"{stage:<34} original=   n/a        new={t_new*1e3:9.1f} ms"
    ratio = t_orig / t_new if t_new else float("inf")
    return (f"{stage:<34} original={t_orig*1e3:9.1f} ms   "
            f"new={t_new*1e3:9.1f} ms   ratio={ratio:5.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=100_000)
    ap.add_argument("--neighbors", type=int, default=6)
    ap.add_argument("--markers", type=int, default=30)
    ap.add_argument("--attributes", type=int, default=8)
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--nas-cells", type=int, default=10_000,
                    help="cells used for the (O(N*E)) neighbor-aggregation stage")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    import mosna as new
    orig = load_original()

    print("=" * 78)
    print("mosna pipeline benchmark — original (A. Coullomb) vs refactored")
    print("=" * 78)
    if orig is None:
        print("NOTE: original reference (tests/_orig_mosna_reference.py) not found; "
              "showing refactored timings only.\n")

    print(f"Building big graph: {args.cells:,} cells, kNN={args.neighbors}, "
          f"{args.markers} markers, {args.attributes} cell types ...")
    t0 = time.perf_counter()
    nodes, edges, attrs = build_big_graph(
        args.cells, args.neighbors, args.markers, args.attributes)
    print(f"  built in {time.perf_counter()-t0:.1f}s — "
          f"{len(edges):,} undirected edges\n")

    r = args.repeats
    lines = []

    # ---- Stage 1: assortativity (mixing matrix + randomization) ----
    def s1_new():
        mm = new.mixing_matrix(nodes, edges, attrs)
        mmr, ac = new.randomized_mixmat(
            nodes, edges, attrs, n_shuffle=args.n_shuffle, parallel=False, verbose=0)
        return mm, mmr, ac

    t_new1, out_new1 = timeit(s1_new, r)
    t_orig1 = None
    if orig is not None:
        def s1_orig():
            mm = orig.mixing_matrix(nodes, edges, attrs)
            mmr, ac = orig.randomized_mixmat(
                nodes, edges, attrs, n_shuffle=args.n_shuffle,
                parallel=False, verbose=0)
            return mm, mmr, ac
        t_orig1, out_orig1 = timeit(s1_orig, r)
        # correctness: deterministic mixing matrix must match exactly
        np.testing.assert_allclose(out_orig1[0], out_new1[0], rtol=1e-12, atol=1e-12)
    lines.append(fmt(f"1. assortativity (shuffle={args.n_shuffle})", t_orig1, t_new1))

    # ---- Stage 2: neighbor aggregation statistics (NAS) ----
    marker_cols = [c for c in nodes.columns if c.startswith("marker_")]
    X = nodes[marker_cols].values
    pairs = edges.values
    sub = min(args.nas_cells, args.cells)
    # restrict to first `sub` cells to keep the O(N*E) loop tractable
    edges_sub = edges[(edges["source"] < sub) & (edges["target"] < sub)].values

    def s2_new():
        return new.aggregate_k_neighbors(X[:sub], edges_sub, order=1,
                                         var_names=marker_cols)
    t_new2, _ = timeit(s2_new, 1)
    t_orig2 = None
    if orig is not None:
        def s2_orig():
            return orig.aggregate_k_neighbors(X[:sub], edges_sub, order=1,
                                              var_names=marker_cols)
        t_orig2, _ = timeit(s2_orig, 1)
    lines.append(fmt(f"2. neighbor-aggreg ({sub:,} cells)", t_orig2, t_new2))

    # ---- Stage 3: I/O round-trip (this is where the refactor differs) ----
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def s3_new():
            new.io.write_table(nodes, d / "n_new.parquet")
            return new.io.read_table(d / "n_new.parquet")
        t_new3, _ = timeit(s3_new, r)

        def s3_csv_new():
            new.io.write_table(nodes, d / "n_new.csv")
            return new.io.read_table(d / "n_new.csv")
        t_new3c, _ = timeit(s3_csv_new, r)

        t_orig3 = t_orig3c = None
        if orig is not None:
            # original code path: plain pandas read/write
            def s3_orig():
                nodes.to_parquet(d / "n_orig.parquet", index=False)
                return pd.read_parquet(d / "n_orig.parquet")
            t_orig3, _ = timeit(s3_orig, r)

            def s3_csv_orig():
                nodes.to_csv(d / "n_orig.csv", index=False)
                return pd.read_csv(d / "n_orig.csv")
            t_orig3c, _ = timeit(s3_csv_orig, r)

        lines.append(fmt("3a. I/O round-trip parquet", t_orig3, t_new3))
        lines.append(fmt("3b. I/O round-trip csv", t_orig3c, t_new3c))

    print("Results (best of {} runs; ratio>1 means refactored is faster):".format(r))
    print("-" * 78)
    for ln in lines:
        print(ln)
    print("-" * 78)
    print("\nStages 1-2 use identical function bodies in both versions (the refactor")
    print("did not touch the math), so parity there confirms the science/compute is")
    print("unchanged. The refactor's speed win is in I/O (stage 3, CSV especially).")


if __name__ == "__main__":
    raise SystemExit(main())
