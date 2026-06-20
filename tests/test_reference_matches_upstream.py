#! /usr/bin/env python3
"""
tests/test_reference_matches_upstream.py
========================================

Option A pinning: prove that the benchmark/equivalence baseline
(``tests/_orig_mosna_reference.py``) really is the upstream
``AlexCoul/mosna`` code, by re-fetching ``mosna/mosna.py`` from GitHub at the
pinned commit and asserting every benchmarked function is byte-identical.

The pinned commit lives in ``benchmarks/UPSTREAM_REF``. This test needs network
access; it is skipped (not failed) when offline so the suite stays green in
sandboxed CI without internet.
"""

import ast
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "tests" / "_orig_mosna_reference.py"
PIN = ROOT / "benchmarks" / "UPSTREAM_REF"

# The functions we benchmark / compare against the original. These must match
# upstream verbatim for the comparison to be meaningful.
BENCHMARKED = [
    "mixing_matrix", "count_edges_undirected", "count_edges_directed",
    "randomized_mixmat", "core_rand_mixmat", "aggregate_k_neighbors",
    "neighbors_k_order", "flatten_neighbors", "neighbors", "attribute_ac",
]


def _read_pin():
    cfg = {}
    for line in PIN.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _func_sources(src):
    tree = ast.parse(src)
    lines = src.splitlines()
    return {
        n.name: "\n".join(lines[n.lineno - 1:n.end_lineno])
        for n in tree.body if isinstance(n, ast.FunctionDef)
    }


def test_reference_matches_pinned_upstream():
    cfg = _read_pin()
    url = cfg["RAW_URL"]
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            upstream = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"no network access to verify upstream pin: {exc}")

    up = _func_sources(upstream)
    ref = _func_sources(REF.read_text())

    mismatched = [
        name for name in BENCHMARKED
        if up.get(name) is None or up.get(name) != ref.get(name)
    ]
    assert not mismatched, (
        f"benchmarked functions differ from upstream {cfg['COMMIT_SHORT']}: "
        f"{mismatched}"
    )
