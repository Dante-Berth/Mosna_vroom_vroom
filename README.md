# MOSNA — Multi‑Omics Spatial Network Analysis (`vroom vroom` edition 🏎️)

**`mosna`** is a Python package for **spatial omics data analysis**: it extracts
clinically relevant features from single-cell spatial measurements
(transcriptomics, proteomics, multiplexed imaging). It reconstructs spatial
networks, computes interaction / neighborhood statistics, and trains predictive
models integrating clinical outcomes.

This repository is a **cleaned-up, modularised and benchmarked** fork of
[AlexCoul/mosna](https://github.com/AlexCoul/mosna). The **scientific logic is
unchanged** — verified bit-for-bit by an equivalence test suite (see
[`tests/`](tests/)). What changed is the *structure* and the *I/O speed*. See
[MODIFICATIONS.md](MODIFICATIONS.md) for the full list.

---

## ⚡ Quick start (install + test, copy-paste)

```bash
# 0. prerequisite: install uv once (https://docs.astral.sh/uv/)
#    curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. create env + install the package and dev/benchmark tools
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e '.[dev,benchmark]'

# 2. check it imports
python -c "import mosna; print('mosna', mosna.__version__, '| gpu:', mosna.gpu_clustering)"

# 3. run the test suite (proves science is unchanged)
pytest -q

# 4. run the I/O speed benchmark
python benchmarks/bench_io.py --rows 200000 --cols 40
```

Or just run the helper script which does all of the above:

```bash
./setup.sh
```

---

## What's new in this fork

| Area | Before | Now |
| ---- | ------ | --- |
| Source layout | one 5 588-line `mosna/mosna.py` | thematic submodules (`assortativity`, `clustering`, `features`, `modeling`, …) |
| Imports | ~50 imports at top of file; GPU/niche deps crash import | centralised in `mosna/_common.py`; optional backends **guarded** |
| Data loading | `pd.read_csv` / `pd.read_parquet` everywhere | fast `mosna.io` layer (Polars/Parquet) — **CSV read ≈13×**, **parquet write ≈4×**, **CSV→Parquet ≈9×** |
| Packaging | `setup.py` | `pyproject.toml` + `uv` |
| Tests | none | equivalence + API tests, I/O benchmark |

Backward compatibility is preserved: `import mosna`, `from mosna import <fn>`,
`from mosna.mosna import <fn>` (legacy) and `from mosna.assortativity import <fn>`
(themed) all work.

---

## Project layout

```text
.
├── mosna/                  # the package
│   ├── __init__.py         # re-exports the full flat API
│   ├── _common.py          # shared imports + optional-backend guards + gpu flag
│   ├── io.py               # ⚡ fast Polars/Parquet I/O helpers
│   ├── utils.py            # small array/data helpers
│   ├── testnets.py         # synthetic test networks
│   ├── preprocessing.py    # transform / aggregate / batch-correct nodes
│   ├── assortativity.py    # mixing matrices & assortativity
│   ├── neighbors.py        # k-order neighbor aggregation statistics
│   ├── features.py         # spatial-omic feature / niche computation
│   ├── clustering.py       # reduction, clustering, cluster post-processing
│   ├── modeling.py         # survival / regression / classification / risk
│   ├── plotting.py         # visualisation helpers
│   ├── screening.py        # parameter screening for niche detection
│   └── mosna.py            # legacy shim (re-exports everything)
├── benchmarks/bench_io.py  # pandas vs Polars/Parquet I/O benchmark
├── tests/                  # equivalence + API tests
├── examples/               # original example notebooks
├── pyproject.toml          # uv / PEP 621 project
└── MODIFICATIONS.md        # detailed changelog of this fork
```

---

## Installation (uv)

```bash
# create the environment
uv venv --python 3.10
source .venv/bin/activate

# core install (editable)
uv pip install -e .

# optional heavy backends, install only what you need:
uv pip install -e '.[spatial]'      # tysserand
uv pip install -e '.[integration]'  # scanorama (batch correction)
uv pip install -e '.[deep]'         # torch / torch_geometric / torchgmm
uv pip install -e '.[gpu]'          # RAPIDS: cupy / cudf / cugraph / cuml
uv pip install -e '.[dev]'          # pytest / ruff
```

If an optional backend is missing, `mosna` still imports fine; the relevant
function raises a clear `ImportError` telling you which extra to install only
when you actually call it.

---

## Fast I/O (`mosna.io`)

The bottleneck in mosna pipelines is loading many wide node/edge tables. The new
`mosna.io` module uses [Polars](https://pola.rs) under the hood but always hands
back ordinary `pandas.DataFrame` objects, so nothing downstream changes:

```python
from mosna import io

nodes = io.read_table("nodes_patient-1.parquet")   # -> pd.DataFrame
io.write_table(nodes, "out.parquet")               # fast parquet write
io.csv_to_parquet("nodes.csv", "nodes.parquet")    # one-off, ~9x faster
```

Internally, mosna's `read_fct = pd.read_parquet/read_csv` pattern was replaced
by `io.get_reader(extension)`, which picks the fastest backend per format
(pandas for already-optimal Parquet reads, Polars for CSV).

### Benchmark

```bash
uv pip install -e '.[benchmark]'
python benchmarks/bench_io.py --rows 200000 --cols 40
```

Representative result (200 000 × 43 table, 20-core machine):

```text
read_parquet     pandas=  61.7 ms   mosna.io=  91.1 ms   speedup= 0.68x
read_csv         pandas= 510.5 ms   mosna.io=  38.3 ms   speedup=13.33x
write_parquet    pandas= 529.6 ms   mosna.io= 136.2 ms   speedup= 3.89x
csv -> parquet   pandas=1135.1 ms   mosna.io= 119.6 ms   speedup= 9.49x
```

The benchmark also asserts `mosna.io` output equals pandas output, so the
speedups never come at the cost of different data. (Full Parquet reads are left
on pandas precisely because Polars offers no win there — see MODIFICATIONS.md.)

---

## Verifying the science is unchanged

```bash
uv pip install -e '.[dev]'
pytest                # runs tests/ (equivalence vs original + API checks)
```

The equivalence suite imports the original monolithic implementation alongside
the refactored package and asserts identical outputs on the test networks and a
battery of pure functions (mixing matrices, assortativity, transforms,
neighbor statistics, z-scores, …).

---

## Analysis pipeline (unchanged science)

| Stage | Description |
| ----- | ----------- |
| 1. Proportions | Quantify global cell-type frequencies |
| 2. Composed variables | Ratios of proportions |
| 3. Assortativity & mixing matrices | Preferential interaction between cell types |
| 4. Neighbors Aggregation Statistics (NAS) | Cell neighborhoods (niches) from local omics |
| 5. Modeling | Survival / response prediction (Cox, logistic, XGBoost) |

See the original [examples notebooks](examples/) for end-to-end usage.

## License

GNU GPLv3 — see [LICENSE](LICENSE).
