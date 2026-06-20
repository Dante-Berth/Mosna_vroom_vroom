How-to guides
=============

Task-oriented recipes. Each assumes you have installed MOSNA
(see :doc:`getting_started`).

.. contents::
   :local:
   :depth: 1


How to load data fast (``mosna.io``)
------------------------------------

The bottleneck in MOSNA pipelines is loading many wide node/edge tables. The
:mod:`mosna.io` layer uses `Polars <https://pola.rs>`_ under the hood but always
hands back ordinary ``pandas.DataFrame`` objects, so nothing downstream changes.

.. code-block:: python

   from mosna import io

   nodes = io.read_table("nodes_patient-1.parquet")   # -> pd.DataFrame
   io.write_table(nodes, "out.parquet")               # fast parquet write
   io.csv_to_parquet("nodes.csv", "nodes.parquet")    # one-off, ~9x faster

   # check whether the fast backend is active
   print(io.have_polars())

``io.get_reader(extension)`` picks the fastest backend per format: pandas for
already-optimal Parquet reads, Polars for CSV. It falls back to pandas
automatically when Polars is unavailable, or when pandas-specific kwargs (e.g.
``dtype=int``) are required.

.. admonition:: Why Parquet reads stay on pandas
   :class: note

   Benchmarking showed Polars helps a lot for CSV and writes but **not** for
   full Parquet reads — pandas/pyarrow is already optimal there and Polars'
   ``to_pandas()`` copy adds overhead. So the slow paths are accelerated and the
   fast one is never regressed.


How to convert a CSV dataset to Parquet
---------------------------------------

Parquet is ~2× smaller on disk and far faster to read. Convert once:

.. code-block:: python

   from mosna import io
   io.csv_to_parquet("nodes.csv", "nodes.parquet")    # ~9x faster than pandas

Then point your pipeline at the ``.parquet`` files.


How to compute a mixing matrix and assortativity
------------------------------------------------

.. code-block:: python

   import mosna

   nodes, edges = mosna.make_high_assort_net()

   # counts of edges between each pair of cell types
   mixmat = mosna.mixing_matrix(nodes, edges, attributes="cell_type")

   # Newman's assortativity coefficient (NumPy 2.x-safe)
   r = mosna.attribute_ac(mixmat)
   print("assortativity:", r)

   # significance vs a permutation null
   rand = mosna.randomized_mixmat(nodes, edges, attributes="cell_type",
                                  n_shuffle=100, random_state=0)
   z = mosna.zscore(mixmat, rand)

.. tip::

   ``randomized_mixmat`` defaults to ``parallel=False`` because each shuffle is
   now ~tens of ms — serial numpy beats process-spawn overhead. Set a
   ``random_state`` for reproducible, bit-identical results between serial and
   parallel runs.


How to build niche features (NAS)
---------------------------------

.. code-block:: python

   import mosna

   nodes, edges = mosna.make_high_assort_net()

   # order-1 mean/std of neighbor markers (vectorised, ~1000x faster)
   feats = mosna.aggregate_k_neighbors(nodes, edges, n_neighbors=1)

   # reduce + cluster into discrete niches
   reducer   = mosna.get_reducer("umap")
   clusterer = mosna.get_clusterer("leiden")
   # ... fit reducer/clusterer on `feats`, then post-process:
   # labels = mosna.relabel_clusters(labels)
   # labels = mosna.merge_clusters_until(labels, ...)


How to model a clinical outcome
-------------------------------

.. code-block:: python

   import mosna

   # survival: search for the best threshold on a feature
   thr = mosna.find_best_survival_threshold(df, variable="niche_3_fraction",
                                            duration_col="time", event_col="event")

   # stepwise regression for variable selection
   model = mosna.stepwise_regression(df, response="event", candidates=[...])

   # gradient boosting
   booster = mosna.train_XGBoost(X, y)

   # relative-risk matrix across niches
   rr = mosna.make_risk_ratio_matrix(df, ...)

See :mod:`mosna.modeling` for the full set of estimators and helpers.


How to run the benchmarks
-------------------------

.. code-block:: bash

   uv pip install -e '.[benchmark]'

   # I/O: pandas vs Polars/Parquet (with a correctness gate)
   python benchmarks/bench_io.py --rows 200000 --cols 40

   # per-kernel compute speedups vs the original (with [OK]/[MISMATCH] flags)
   python benchmarks/bench_compute.py --cells 100000 --attributes 10

   # end-to-end original vs refactored on a big graph
   python benchmarks/bench_pipeline.py

Every benchmark asserts the optimised output matches the original *before*
reporting a speedup, so speed never costs correctness.
