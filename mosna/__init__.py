#! /usr/bin/env python3
"""
mosna — Multi-Omics Spatial Network Analysis
============================================

The library was historically a single ~5500 line module (``mosna.mosna``).
It is now organised into thematic submodules; this ``__init__`` re-exports the
full public API so that existing code keeps working unchanged::

    import mosna
    from mosna import groups_assort_mixmat, get_clusterer       # flat API
    from mosna.assortativity import mixing_matrix               # themed API
    from mosna.mosna import mixing_matrix                       # legacy shim

Submodules (imported in dependency order):

- :mod:`mosna._common`       shared imports & optional-backend handling
- :mod:`mosna.utils`         small array/data helpers
- :mod:`mosna.testnets`      synthetic test networks
- :mod:`mosna.preprocessing` transform / aggregate / batch-correct nodes
- :mod:`mosna.assortativity` mixing matrices & assortativity
- :mod:`mosna.neighbors`     k-order neighbor aggregation statistics
- :mod:`mosna.features`      spatial-omic feature / niche computation
- :mod:`mosna.clustering`    reduction, clustering & cluster post-processing
- :mod:`mosna.modeling`      survival / regression / classification / risk
- :mod:`mosna.plotting`      visualisation helpers
- :mod:`mosna.screening`     parameter screening for niche detection
- :mod:`mosna.io`            fast Polars/Parquet I/O helpers
"""

from mosna._common import gpu_clustering  # noqa: F401

# Import order follows the (acyclic) inter-module dependency graph.
from mosna.utils import *          # noqa: F401,F403
from mosna.testnets import *       # noqa: F401,F403
from mosna.preprocessing import *  # noqa: F401,F403
from mosna.assortativity import *  # noqa: F401,F403
from mosna.neighbors import *      # noqa: F401,F403
from mosna.features import *       # noqa: F401,F403
from mosna.clustering import *     # noqa: F401,F403
from mosna.modeling import *       # noqa: F401,F403
from mosna.plotting import *       # noqa: F401,F403
from mosna.screening import *      # noqa: F401,F403

from mosna import io  # noqa: F401  fast Polars/Parquet helpers

__version__ = "0.1.0"
