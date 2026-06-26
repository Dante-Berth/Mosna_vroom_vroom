# MOSNA at a glance

## What problem does it solve?

Modern spatial-omics assays (multiplexed imaging, spatial transcriptomics /
proteomics) measure many molecular markers for **each individual cell**, together
with the cell's **spatial location**. MOSNA turns that raw per-cell table into
**clinically interpretable features** by reasoning about the *spatial network* of
cells:

1. Which cell types preferentially sit next to which (assortativity / mixing)?
2. What does each cell's local neighborhood look like (niches)?
3. Do those spatial features predict a clinical outcome (survival, response)?

The library reconstructs a spatial graph, computes neighborhood statistics on it,
and feeds the resulting features into survival / classification models.

## How the package is organised

This fork splits the original 5 588-line monolith into thematic submodules. Each
does one job; the flat `import mosna` API still re-exports everything for
backward compatibility.

| Module | Responsibility |
| --- | --- |
| [`mosna.io`][mosna.io] | ⚡ Fast Polars / Parquet I/O helpers (always return `pandas.DataFrame`). |
| [`mosna.utils`][mosna.utils] | Small array / data helpers (`to_numpy`, `renormalize`). |
| [`mosna.testnets`][mosna.testnets] | Synthetic test networks (`make_*_net`) used in tests and demos. |
| [`mosna.preprocessing`][mosna.preprocessing] | Transform / aggregate / batch-correct node tables. |
| [`mosna.assortativity`][mosna.assortativity] | Mixing matrices and assortativity statistics. |
| [`mosna.neighbors`][mosna.neighbors] | k-order neighbor aggregation statistics (NAS). |
| [`mosna.features`][mosna.features] | Spatial-omic feature / niche computation. |
| [`mosna.clustering`][mosna.clustering] | Dimensionality reduction, clustering, cluster post-processing. |
| [`mosna.modeling`][mosna.modeling] | Survival / regression / classification / risk statistics. |
| [`mosna.plotting`][mosna.plotting] | Visualisation helpers. |
| [`mosna.screening`][mosna.screening] | Parameter screening for niche detection. |

!!! note

    `mosna/_common.py` centralises the shared imports and **guards** optional
    backends (`tysserand`, `scanorama`, `torchgmm`, the RAPIDS GPU stack…).
    A missing backend no longer crashes `import mosna`; it raises a clear
    `ImportError` only if/when you actually call the function that needs it.

## What makes this fork fast

Three things, each verified to leave results unchanged:

* **Fast I/O** ([`mosna.io`][mosna.io]): Polars/Parquet for the slow paths — CSV read
  ≈ **13×**, parquet write ≈ **4×**, CSV→Parquet ≈ **9×**. Parquet *reads* stay
  on pandas because Polars offers no win there.
* **Vectorised assortativity**: `mixing_matrix` rewritten with boolean numpy
  algebra (and a `bincount` fast path for one-hot cell types) — up to **~210×**,
  bit-identical.
* **Vectorised neighbor aggregation**: order-1 mean/std via sparse matrix
  products instead of a per-cell Python loop — up to **~1000×**, identical to
  ~1e-15.

See [How-to guides](how_to_guides.md) for usage and
[Theoretical concepts](theoretical_concepts.md) for the math.

## Backward compatibility

Every existing import style keeps working:

```python
import mosna
from mosna import mixing_matrix                 # flat (re-exported)
from mosna.assortativity import mixing_matrix   # themed (new)
from mosna.mosna import mixing_matrix           # legacy shim
```
