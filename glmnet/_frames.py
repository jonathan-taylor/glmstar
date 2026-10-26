"""
Data frames other than pandas: polars, pyarrow, modin, cuDF, DuckDB...

The estimators work on pandas DataFrames and numpy arrays; a data frame
of another library is converted to pandas at the boundary (`as_pandas`),
through narwhals and numpy (no pyarrow needed for polars). The outputs
(coefficients, summaries, tables) stay pandas.
"""

import numpy as np
import pandas as pd
import scipy.sparse

import narwhals.stable.v2 as nw


def as_pandas(obj):
    """
    `obj` as pandas, if it is a data frame or series of another library.

    numpy arrays, sparse matrices, pandas objects, None and anything
    narwhals does not know (lists, `DesignSpec`...) are returned unchanged.
    A lazy frame (polars `LazyFrame`, a DuckDB relation) is collected.

    A frame of numeric columns becomes one 2d array (a single copy, which
    pandas wraps without another); otherwise the columns are converted one
    at a time: an `Enum` keeps its levels and their order, a `Categorical`
    becomes a pandas categorical (its levels sorted, as `astype('category')`
    does), strings become objects. Nulls are NaN (None in object columns).
    The index is a `RangeIndex`.
    """
    if (obj is None or isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series))
        or scipy.sparse.issparse(obj)):
        return obj
    native = nw.from_native(obj, pass_through=True, allow_series=True)
    if native is obj:
        return obj
    if isinstance(native, nw.LazyFrame):
        native = native.collect()
    if isinstance(native, nw.Series):
        return pd.Series(_column(native), name=native.name)
    if all(dtype.is_numeric() for dtype in native.schema.values()):
        return pd.DataFrame(native.to_numpy(), columns=native.columns, copy=False)
    return pd.DataFrame({name: _column(native[name]) for name in native.columns},
                        copy=False)


def _column(series):
    """One narwhals series as a numpy array or pandas categorical."""
    dtype = series.dtype
    if isinstance(dtype, nw.Enum):
        return pd.Categorical(series.to_numpy(), categories=list(dtype.categories))
    if dtype == nw.Categorical:
        return pd.Categorical(series.to_numpy())
    return series.to_numpy()
