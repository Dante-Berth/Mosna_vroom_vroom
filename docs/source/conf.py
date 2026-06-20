# Configuration file for the Sphinx documentation builder.
#
# MOSNA — Multi-Omics Spatial Network Analysis ("vroom vroom" edition 🏎️)
# Documentation for the accelerated fork: https://github.com/Dante-Berth/Mosna_vroom_vroom
#
# Full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import os
import sys

# Make the `mosna` package importable by autodoc (repo root is three levels up
# from this file: docs/source/conf.py -> repo root).
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "MOSNA (vroom vroom edition)"
copyright = "2026, Alexis Coullomb (original science); Dante-Berth (accelerated fork)"
author = "Alexis Coullomb · accelerated fork by Dante-Berth"

try:
    release = importlib.metadata.version("mosna")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy / Google style docstrings
    "sphinx.ext.viewcode",      # [source] links
    "sphinx.ext.intersphinx",
    "myst_parser",              # lets us include the Markdown files (README etc.)
    "nbsphinx",                 # render the example notebooks
]

autosummary_generate = True
add_module_names = False  # show `mixing_matrix`, not `mosna.assortativity.mixing_matrix`

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Heavy / optional / GPU backends are mocked so RTD can import `mosna` and read
# docstrings without installing the full scientific stack. These mirror the
# guarded optional backends in mosna/_common.py plus the core compiled deps.
autodoc_mock_imports = [
    "numpy", "pandas", "polars", "pyarrow", "scipy", "sklearn",
    "skimage", "statsmodels", "matplotlib", "seaborn", "colorcet",
    "tqdm", "joblib", "igraph", "leidenalg", "anndata", "lifelines",
    "sksurv", "xgboost", "composition_stats", "umap", "hdbscan",
    "fastcluster", "dask", "openpyxl", "odf",
    # optional / niche backends:
    "tysserand", "scanorama", "torch", "torch_geometric", "torchgmm",
    "gudhi", "napari", "cupy", "cudf", "cugraph", "cuml",
]

# When the scientific stack is mocked, symbols like ``UndefinedMetricWarning``
# (used as a ``warnings.filterwarnings(category=...)`` argument in
# ``mosna._common``) become Mock objects, which trips an internal assertion in
# the stdlib ``warnings`` module ("category must be a class"). Make
# ``filterwarnings`` tolerant of non-class categories so importing ``mosna`` for
# autodoc never hard-fails under mocking. This only affects the docs build.
import warnings as _warnings

_orig_filterwarnings = _warnings.filterwarnings


def _safe_filterwarnings(*args, **kwargs):
    try:
        return _orig_filterwarnings(*args, **kwargs)
    except (AssertionError, TypeError):
        return None


_warnings.filterwarnings = _safe_filterwarnings

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# nbsphinx: do not execute notebooks at build time (they need data + heavy deps).
nbsphinx_execute = "never"
# The notebooks declare an ``ipython3`` cell lexer; without IPython installed
# Pygments cannot find it and emits hundreds of warnings. Fall back to plain
# ``python3`` highlighting for code cells.
nbsphinx_codecell_lexer = "python3"
highlight_language = "python3"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The original example notebooks contain Markdown headings that jump levels
# (e.g. a ``^^^`` sub-sub-heading without a parent), which docutils reports as
# "Title level inconsistent". We render the notebooks verbatim and do not edit
# their scientific content, so suppress that one cosmetic category — otherwise
# RTD's fail_on_warning would reject the build over upstream notebook style.
suppress_warnings = ["docutils"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "MOSNA vroom vroom 🏎️"
