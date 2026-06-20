#! /usr/bin/env python3
"""
mosna.io — fast Polars / Parquet I/O helpers
============================================

This module provides drop-in faster replacements for the ``pandas`` read/write
calls used throughout ``mosna``. The functions here use `Polars
<https://pola.rs>`_ (a multi-threaded, Arrow-backed dataframe engine) for the
heavy lifting and then hand back **ordinary** :class:`pandas.DataFrame`
objects, so the rest of the scientific pipeline is byte-for-byte unaffected —
only the loading/saving step is accelerated.

Design goals
------------
1. **Same science.** Every function returns/writes the exact same data a
   ``pandas`` call would. We only change *how fast* the bytes move.
2. **Graceful fallback.** If Polars is not installed, we transparently fall
   back to pandas so nothing breaks.
3. **Parquet first.** Parquet is columnar and compressed; for the kind of
   wide numeric node/edge tables mosna handles it is both smaller on disk and
   much faster to load than CSV. Helpers are provided to convert legacy CSVs.

Typical usage
-------------
>>> from mosna import io
>>> nodes = io.read_table("nodes_patient-1.parquet")      # fast, -> pd.DataFrame
>>> io.write_table(nodes, "out.parquet")                  # fast parquet write
>>> io.csv_to_parquet("nodes.csv", "nodes.parquet")       # one-off conversion
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

try:
    import polars as pl
    _HAVE_POLARS = True
except ImportError:  # pragma: no cover
    pl = None
    _HAVE_POLARS = False

PathLike = Union[str, Path]

__all__ = [
    "have_polars",
    "read_parquet",
    "read_csv",
    "read_table",
    "write_parquet",
    "write_table",
    "csv_to_parquet",
    "get_reader",
]


def have_polars() -> bool:
    """Whether the Polars acceleration backend is available."""
    return _HAVE_POLARS


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def read_parquet(
    path: PathLike,
    columns: Optional[Sequence[str]] = None,
    use_polars: bool = False,
) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame.

    Defaults to pandas/pyarrow: benchmarking showed that for full Parquet reads
    pandas is already optimal and the Polars ``to_pandas()`` copy adds overhead
    (see ``benchmarks/bench_io.py``). Polars helps for CSV and writes, not
    full Parquet reads, so the default here avoids a regression. Pass
    ``use_polars=True`` to force the Polars path.
    The returned object is always a ``pandas.DataFrame``.
    """
    if use_polars and _HAVE_POLARS:
        df = pl.read_parquet(path, columns=list(columns) if columns else None)
        return df.to_pandas()
    return pd.read_parquet(path, columns=columns)


def read_csv(
    path: PathLike,
    columns: Optional[Sequence[str]] = None,
    use_polars: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame, accelerated with Polars.

    Extra ``kwargs`` are forwarded to the pandas fallback for full
    compatibility (e.g. ``dtype=int`` as used in the original code).
    """
    if use_polars and _HAVE_POLARS and not kwargs:
        df = pl.read_csv(path, columns=list(columns) if columns else None)
        return df.to_pandas()
    pdf = pd.read_csv(path, **kwargs)
    if columns is not None:
        pdf = pdf[list(columns)]
    return pdf


def read_table(
    path: PathLike,
    columns: Optional[Sequence[str]] = None,
    use_polars: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a table, dispatching on file extension (``.parquet`` / ``.csv``)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".parquet", ".pq"):
        return read_parquet(path, columns=columns, use_polars=use_polars)
    if suffix in (".csv", ".txt"):
        return read_csv(path, columns=columns, use_polars=use_polars, **kwargs)
    raise ValueError(f"Unsupported table extension: {suffix!r}")


def get_reader(extension: str, use_polars: Optional[bool] = None):
    """Return a ``read_fct`` matching mosna's ``pd.read_parquet``/``read_csv``.

    Drop-in for the ``read_fct = pd.read_parquet`` pattern found throughout the
    original code::

        read_fct = io.get_reader(extension)
        nodes = read_fct(path)

    By default it picks the fastest backend per format: pandas/pyarrow for
    Parquet (already optimal) and Polars for CSV (~10x faster). Pass
    ``use_polars`` to override.
    """
    ext = extension.lstrip(".").lower()
    if ext in ("parquet", "pq"):
        up = False if use_polars is None else use_polars
        return lambda p, **kw: read_parquet(p, use_polars=up, **kw)
    if ext == "csv":
        up = True if use_polars is None else use_polars
        return lambda p, **kw: read_csv(p, use_polars=up, **kw)
    raise ValueError(f"Unsupported extension: {extension!r}")


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def write_parquet(
    df: pd.DataFrame,
    path: PathLike,
    index: bool = False,
    use_polars: bool = True,
    compression: str = "zstd",
) -> None:
    """Write a pandas DataFrame to Parquet, accelerated with Polars.

    Mirrors the original ``df.to_parquet(path, index=False)`` calls. When
    ``index=True`` we fall back to pandas so the index is preserved exactly.
    """
    if use_polars and _HAVE_POLARS and not index:
        pl.from_pandas(df).write_parquet(path, compression=compression)
    else:
        df.to_parquet(path, index=index)


def write_table(
    df: pd.DataFrame,
    path: PathLike,
    index: bool = False,
    use_polars: bool = True,
) -> None:
    """Write a table, dispatching on extension (``.parquet`` / ``.csv``)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".parquet", ".pq"):
        write_parquet(df, path, index=index, use_polars=use_polars)
    elif suffix in (".csv", ".txt"):
        if use_polars and _HAVE_POLARS and not index:
            pl.from_pandas(df).write_csv(path)
        else:
            df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported table extension: {suffix!r}")


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #

def csv_to_parquet(
    csv_path: PathLike,
    parquet_path: Optional[PathLike] = None,
    compression: str = "zstd",
    use_polars: bool = True,
) -> Path:
    """Convert a CSV file to Parquet (much faster & smaller for reuse).

    Returns the path to the written Parquet file.
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    parquet_path = Path(parquet_path)

    if use_polars and _HAVE_POLARS:
        pl.read_csv(csv_path).write_parquet(parquet_path, compression=compression)
    else:
        pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
    return parquet_path
