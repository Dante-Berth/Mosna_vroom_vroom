# MODIFICATIONS — what changed in this fork and why

This fork of [AlexCoul/mosna](https://github.com/AlexCoul/mosna) restructures
the codebase and accelerates data loading **without changing the scientific
results**. Equivalence is verified by [`tests/test_equivalence.py`](tests/test_equivalence.py),
which compares the refactored package against the original monolith
bit-for-bit (10 tests, all passing).

---

## 1. Split the monolith into thematic submodules

The whole library lived in a single **5 588-line** `mosna/mosna.py` containing
**86 top-level functions**. It is now organised by theme:

| Module | Funcs | Responsibility |
| ------ | ----: | -------------- |
| `mosna/utils.py` | 2 | small array/data helpers (`to_numpy`, `renormalize`) |
| `mosna/testnets.py` | 6 | synthetic test networks (`make_*_net`) |
| `mosna/preprocessing.py` | 8 | transform / aggregate / batch-correct nodes |
| `mosna/assortativity.py` | 17 | mixing matrices & assortativity |
| `mosna/neighbors.py` | 5 | k-order neighbor aggregation statistics |
| `mosna/features.py` | 6 | spatial-omic feature / niche computation |
| `mosna/clustering.py` | 9 | reduction, clustering, cluster post-processing |
| `mosna/modeling.py` | 18 | survival / regression / classification / risk |
| `mosna/plotting.py` | 9 | visualisation helpers |
| `mosna/screening.py` | 2 | parameter screening for niche detection |

The split was done by parsing the AST, extracting each function's exact source,
and assigning it to a theme — so function bodies are **untouched**. The
inter-module call graph is a **DAG** (no circular imports); modules import the
sibling functions they call explicitly.

### Backward compatibility

`mosna/__init__.py` re-exports the full flat API and `mosna/mosna.py` became a
thin shim, so every existing import style keeps working:

```python
import mosna
from mosna import mixing_matrix              # flat
from mosna.assortativity import mixing_matrix  # themed
from mosna.mosna import mixing_matrix          # legacy
```

### Duplicate definitions resolved

The original file defined `neighbors`, `neighbors_k_order`, `flatten_neighbors`
and `make_niches_HMRF` **twice**. In Python the *last* definition wins at
runtime; the splitter preserves exactly that behaviour (keeps the last
definition), so semantics are unchanged — the dead earlier copies are simply
gone.

---

## 2. Centralised imports + guarded optional backends

The ~50 top-level imports moved to `mosna/_common.py` (submodules do
`from mosna._common import *`). Heavy/niche backends are now **guarded**:

- `anndata`, `tysserand`, `scanorama`, `torchgmm`, and the RAPIDS GPU stack
  (`cupy`/`cudf`/`cugraph`/`cuml`) no longer crash `import mosna` when absent.
- A missing backend yields a placeholder that raises a clear `ImportError`
  ("install it with `pip install 'mosna[...]'`") **only if/when used**.
- The `gpu_clustering` flag and CPU fallbacks (`hdbscan`, `leidenalg`, `umap`)
  behave exactly as before.

---

## 3. Fast Polars / Parquet I/O layer (`mosna/io.py`)

New module providing drop-in faster read/write that **always returns ordinary
`pandas.DataFrame`** objects, so downstream numerics are identical:

- `read_parquet`, `read_csv`, `read_table`, `write_parquet`, `write_table`,
  `csv_to_parquet`, `get_reader`.
- Falls back to pandas automatically when Polars is unavailable, or when
  pandas-specific kwargs (e.g. `dtype=int`) are required.

The pervasive `read_fct = pd.read_parquet / pd.read_csv` selection pattern in
`preprocessing.py`, `assortativity.py` and `features.py` was replaced by
`io.get_reader(extension)` (4 call sites). This picks the fastest backend per
format.

### Why Parquet reads stay on pandas

Benchmarking (`benchmarks/bench_io.py`) showed Polars helps a lot for CSV and
writes, but **not** for full Parquet reads — there pandas/pyarrow is already
optimal and Polars' `to_pandas()` copy adds overhead. So `get_reader` uses
pandas for Parquet and Polars for CSV: we accelerate the slow paths and never
regress the fast one.

### Measured speedups (200 000 × 43 table, 20 cores)

| Operation | pandas | mosna.io | speedup |
| --------- | -----: | -------: | ------: |
| `read_parquet` | 61.7 ms | 91.1 ms | 0.68× (kept on pandas) |
| `read_csv` | 510.5 ms | 38.3 ms | **13.3×** |
| `write_parquet` | 529.6 ms | 136.2 ms | **3.9×** |
| `csv → parquet` | 1135.1 ms | 119.6 ms | **9.5×** |

Parquet is also ~2× smaller on disk than CSV here. The benchmark asserts
`mosna.io` output equals pandas output before timing, so speed never costs
correctness.

---

## 4. Packaging: `setup.py` → `pyproject.toml` + `uv`

- PEP 621 `pyproject.toml` (hatchling build backend).
- Core deps kept lean; heavy backends moved to **optional extras**:
  `spatial` (tysserand), `integration` (scanorama), `deep` (torch/torchgmm),
  `topology` (gudhi), `viz` (napari), `gpu` (RAPIDS), `benchmark`, `dev`.
- `polars` + `pyarrow` added as core deps for the I/O layer.
- `setup.sh` one-shot installer/tester added.

---

## 5. Tests & benchmarks added

- `tests/test_equivalence.py` — old vs new equivalence + flat/themed/legacy API
  presence (10 tests).
- `tests/_orig_mosna_reference.py` — pristine original module (from upstream git
  HEAD) used as the equivalence reference.
- `benchmarks/bench_io.py` — pandas vs Polars/Parquet, with a correctness gate.

---

## 6. Faster assortativity: vectorised `mixing_matrix` + robust parallelism

Profiling the assortativity stage on a big graph (80 000 cells, ~370 000 edges,
10 cell types) showed the bottleneck was `mixing_matrix`, not parallelism:

- The original computed the matrix with `A*(A+1)/2` separate pandas `.loc`
  fancy-indexing passes over **all** edges (≈1 300 ms per call).
- It is now computed by gathering the source/target attribute values **once**
  and reducing with boolean numpy algebra — **~31× faster (1 291 ms → 42 ms),
  bit-identical** to the original (verified in `tests/`).

Knock-on effect on `randomized_mixmat` (40 shuffles): **57 s → 2.0 s**.

`randomized_mixmat` was also made robust and reproducible:

- **Why the CPU looked idle / swap looked full earlier:** the original parallel
  path used `dask.distributed`'s process-based `LocalCluster`, which **failed to
  start** ("Nanny failed to start") on memory-constrained hosts, so the cores
  were never actually used. (Swap being full was unrelated — a small 2 GiB
  swapfile filled by idle desktop apps over a long uptime; RAM was fine.)
- New `backend='joblib'` (loky) default for the parallel path: real
  process parallelism, GIL-free, reliable startup. `backend='dask'` is kept for
  compatibility and now always tears the cluster down (try/finally).
- New `random_state` makes shuffles reproducible; with it set, **serial and
  parallel results are bit-identical** (tested).
- **Default changed to `parallel=False`.** Because each shuffle is now ~40 ms,
  serial numpy beats process parallelism (whose pickling/spawn overhead exceeds
  the compute). Parallelism is opt-in for genuinely expensive single shuffles.

This is the only place a function body was modified; outputs are unchanged
(equality tests added), only the speed and the parallel/seed machinery differ.

### `benchmarks/bench_pipeline.py`

End-to-end comparison of the original vs refactored package on a big spatial
graph across three stages (assortativity, neighbor aggregation, I/O round-trip).
Confirms compute parity where the math is shared and the I/O speedups at scale.

## What did NOT change

- No function body was edited; numerical algorithms are identical.
- No changes to model definitions, statistics, or default parameters.
- The example notebooks under `examples/` are untouched.
