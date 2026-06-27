#! /usr/bin/env python3
"""
mosna._common
=============

Shared imports, optional-backend handling and module-level state used across
all ``mosna`` submodules.

Historically ``mosna`` was a single 5500+ line module whose top of file held
~50 imports. Those imports are centralised here so the thematic submodules can
simply do ``from mosna._common import *`` and stay focused on logic.

Optional / heavy backends (GPU stack, ``scanorama``, ``tysserand``,
``torchgmm`` ...) are imported lazily/guarded so that importing ``mosna`` never
fails just because one optional dependency is missing. The relevant functions
raise a clear error only when the missing backend is actually used.
"""

import os
import re
import warnings
from copy import deepcopy
from functools import partial
from itertools import combinations
from pathlib import Path
from time import time
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union,
)
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

from scipy.spatial import cKDTree
from scipy import stats
from scipy.stats import ttest_ind     # Welch's t-test
from scipy.stats import mannwhitneyu  # Mann-Whitney rank test
from scipy.stats import ks_2samp      # Kolmogorov-Smirnov statistic

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn import linear_model
from sklearn.model_selection import (
    GridSearchCV, KFold, train_test_split, GroupKFold,
)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import FitFailedWarning, UndefinedMetricWarning
from sklearn import metrics
from sklearn.decomposition._pca import PCA as PCA_type
from sklearn.decomposition import PCA

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import fdrcorrection

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import inv_normal_cdf

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

import xgboost
import composition_stats as cs
import igraph as ig

from dask.distributed import Client, LocalCluster, progress
from dask import delayed
import dask

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# --------------------------------------------------------------------------- #
# Optional / heavy backends — guarded so importing mosna never hard-fails.
# --------------------------------------------------------------------------- #

def _missing(name, extra):
    """Return a placeholder whose use raises an informative ImportError."""
    class _Missing:
        def __getattr__(self, _attr):
            raise ImportError(
                f"Optional dependency '{name}' is required for this feature. "
                f"Install it with: pip install 'mosna[{extra}]'"
            )

        def __call__(self, *_a, **_k):
            raise ImportError(
                f"Optional dependency '{name}' is required for this feature. "
                f"Install it with: pip install 'mosna[{extra}]'"
            )
    return _Missing()


try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = _missing("anndata", "spatial")

try:
    from tysserand import tysserand as ty
except ImportError:  # pragma: no cover
    ty = _missing("tysserand", "spatial")

try:
    import scanorama
except ImportError:  # pragma: no cover
    scanorama = _missing("scanorama", "integration")

try:
    from torchgmm.bayes import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = _missing("torchgmm", "deep")


# GPU clustering stack (RAPIDS). Fall back to CPU implementations otherwise.
try:
    import cupy as cp
    import cugraph
    import cudf
    from cuml import HDBSCAN
    from cuml.cluster.hdbscan import all_points_membership_vectors
    gpu_clustering = True
except Exception:  # broad: GPU libs raise many error types when unavailable
    cp = _missing("cupy", "gpu")
    cugraph = _missing("cugraph", "gpu")
    cudf = _missing("cudf", "gpu")
    from hdbscan import HDBSCAN
    from hdbscan import all_points_membership_vectors
    import leidenalg as la
    gpu_clustering = False

# UMAP is always imported from the CPU package (matches original behaviour).
from umap import UMAP
