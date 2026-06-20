Theoretical concepts
====================

This page explains the analysis pipeline and the statistics behind it. The math
is identical to the original MOSNA; this fork only changes *how fast* it is
computed. Where a kernel was rewritten for speed, the new code is asserted
**bit-identical** (or identical to ~1e-15) against the original.

The analysis pipeline
---------------------

.. list-table::
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - Stage
     - Description
   * - 1
     - Proportions
     - Quantify global cell-type frequencies per sample.
   * - 2
     - Composed variables
     - Ratios of proportions (interpretable composite features).
   * - 3
     - Assortativity & mixing matrices
     - Preferential interaction between cell types in the spatial graph.
   * - 4
     - Neighbors Aggregation Statistics (NAS)
     - Cell neighborhoods (niches) summarised from local omics.
   * - 5
     - Modeling
     - Survival / response prediction (Cox, logistic, XGBoost).

Spatial networks
----------------

Each sample becomes a graph: **nodes** are cells (with coordinates + marker
values) and **edges** connect spatially adjacent cells. MOSNA consumes a
``nodes`` table and an ``edges`` table; synthetic generators in
:mod:`mosna.testnets` (``make_high_assort_net``, ``make_random_graph_2libs`` …)
produce networks with known structure for testing and demos.

Mixing matrices & assortativity
-------------------------------

The **mixing matrix** :math:`M` counts edges between attribute categories
(e.g. cell types): :math:`M_{ij}` is the number (or fraction) of edges between a
cell of type *i* and a cell of type *j*. It captures whether like sits with like
(assortative) or unlike with unlike (disassortative).

**Newman's assortativity coefficient** (Newman 2003, Eq. 2) summarises the
normalised mixing matrix :math:`e` into a single scalar:

.. math::

   r = \frac{\operatorname{Tr}(e) - \lVert e^2 \rVert}{1 - \lVert e^2 \rVert}

implemented in :func:`mosna.attribute_ac`. :math:`r = 1` is perfectly
assortative, :math:`r < 0` disassortative.

.. admonition:: How the speedup keeps results identical
   :class: note

   * **General case**: instead of ``A*(A+1)/2`` pandas ``.loc`` passes over all
     edges, the source/target attribute values are gathered once and reduced with
     boolean numpy algebra.
   * **One-hot cell types**: the whole matrix is a single ``np.bincount`` of the
     ``(src_code, tgt_code)`` edge pairs.
   * Both are verified bit-identical to the original ``mixing_matrix``.
   * **NumPy 2.x fix**: :func:`mosna.attribute_ac` replaces ``np.asmatrix`` /
     ``float(1×1 matrix)`` (a hard ``TypeError`` on NumPy ≥ 2.3) with
     ``(M @ M).sum()`` and ``np.trace(M)`` — mathematically identical, verified
     over 2000 random matrices.

Randomized mixing matrices (significance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test whether observed mixing is more than chance, :func:`mosna.randomized_mixmat`
permutes the attribute labels many times and recomputes the matrix, giving a null
distribution (and a z-score via :func:`mosna.zscore`).

In this fork the serial path shuffles the integer attribute **codes** and reduces
with one ``bincount`` per shuffle (~9× faster). Guarantees: each shuffle equals
``mixing_matrix`` on the same permutation, results are reproducible for a fixed
``random_state``, and the null distribution is statistically equivalent to the
original (all tested). A robust ``joblib`` (loky) parallel backend replaces the
fragile process-based dask cluster.

Neighbors Aggregation Statistics (NAS)
--------------------------------------

A cell's **niche** is described by aggregating the omics of its neighbors. For
each cell, MOSNA collects neighbors up to order *k* (:func:`mosna.neighbors_k_order`),
flattens them (:func:`mosna.flatten_neighbors`), and computes summary statistics
(mean, std, …) of the marker values across the neighborhood
(:func:`mosna.aggregate_k_neighbors`).

For the common case (**order 1**, mean/std) the closed neighborhood is the
adjacency matrix with self-loops, so the statistics come from sparse matrix
products:

.. math::

   \text{mean} = \frac{A \cdot X}{\deg}, \qquad
   \text{std}  = \sqrt{\frac{A \cdot X^2}{\deg} - \text{mean}^2}

This replaces the per-cell Python loop with vectorised sparse algebra
(**~1000×** faster, identical to ~1e-15, including isolated nodes). Higher
orders and custom statistics fall back to the original exact loop.

Niches & clustering
-------------------

The NAS feature matrix is reduced (PCA / UMAP via :func:`mosna.get_reducer`) and
clustered (Leiden / HDBSCAN via :func:`mosna.get_clusterer`) to define discrete
**niche labels**. Post-processing (:func:`mosna.relabel_clusters`,
:func:`mosna.merge_clusters`) cleans up the labeling. Alternative niche backends
(STAGATE, SCANIT, HMRF) live in :mod:`mosna.features`.

Modeling clinical outcomes
--------------------------

The per-sample niche/assortativity features feed survival and classification
models in :mod:`mosna.modeling`: Cox proportional hazards and threshold search
(:func:`mosna.find_best_survival_threshold`), stepwise logistic / linear
regression (:func:`mosna.stepwise_regression`), gradient boosting
(:func:`mosna.train_XGBoost`), and relative-risk statistics
(:func:`mosna.relative_risk`, :func:`mosna.make_risk_ratio_matrix`).

References
----------

* M. E. J. Newman, *Mixing patterns in networks*, Phys. Rev. E 67, 026126 (2003).
* Original MOSNA methodology — see the paper linked from the repository README.
