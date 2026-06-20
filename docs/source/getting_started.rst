Getting started
===============

This page gets you from zero to a working, *verified* MOSNA install.

Prerequisites
-------------

* Python **3.10** or **3.11**.
* `uv <https://docs.astral.sh/uv/>`_ (recommended) — a fast Python package
  manager. Install it once::

      curl -LsSf https://astral.sh/uv/install.sh | sh

Install (copy-paste)
--------------------

.. code-block:: bash

   # 1. create env + install the package and dev/benchmark tools
   uv venv --python 3.10
   source .venv/bin/activate
   uv pip install -e '.[dev,benchmark]'

   # 2. check it imports
   python -c "import mosna; print('mosna', mosna.__version__, '| gpu:', mosna.gpu_clustering)"

   # 3. run the test suite (proves the science is unchanged)
   pytest -q

   # 4. run the I/O speed benchmark
   python benchmarks/bench_io.py --rows 200000 --cols 40

Or run the helper script which does all of the above::

   ./setup.sh

Optional heavy backends
-----------------------

Install only what you need; the package imports fine without them.

.. code-block:: bash

   uv pip install -e '.[spatial]'      # tysserand (network reconstruction)
   uv pip install -e '.[integration]'  # scanorama (batch correction)
   uv pip install -e '.[deep]'         # torch / torch_geometric / torchgmm
   uv pip install -e '.[topology]'     # gudhi
   uv pip install -e '.[viz]'          # napari
   uv pip install -e '.[gpu]'          # RAPIDS: cupy / cudf / cugraph / cuml

If an optional backend is missing, the relevant function raises a clear
``ImportError`` telling you which extra to install — only when you call it.

Verify the science is unchanged
-------------------------------

.. code-block:: bash

   uv pip install -e '.[dev]'
   pytest        # tests/ — equivalence vs the original + API presence checks

The equivalence suite imports the original monolithic implementation alongside
the refactored package and asserts identical outputs on the test networks and a
battery of pure functions (mixing matrices, assortativity, transforms,
neighbor statistics, z-scores …). See :doc:`theoretical_concepts` for what is
compared and why it stays identical.

Your first analysis
-------------------

A minimal end-to-end taste using a synthetic test network:

.. code-block:: python

   import mosna

   # 1. build a synthetic spatial network with two preferentially-mixing groups
   nodes, edges = mosna.make_high_assort_net()

   # 2. assortativity: how do the cell types preferentially interact?
   mixmat = mosna.mixing_matrix(nodes, edges, attributes="cell_type")
   print(mixmat)

   # 3. neighbor aggregation statistics (niche descriptors)
   features = mosna.aggregate_k_neighbors(nodes, edges, n_neighbors=1)
   print(features.head())

.. tip::

   For real data, load your node/edge tables with the fast I/O layer::

       from mosna import io
       nodes = io.read_table("nodes_patient-1.parquet")   # -> pd.DataFrame

   See :doc:`how_to_guides` for the full recipe set, and the rendered
   :doc:`examples` notebooks for complete real-data pipelines.
