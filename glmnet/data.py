"""
glmnet.data
------------
Data utilities for GLMNet models.

This module provides functions to generate synthetic datasets for various
GLMNet estimator classes (LogNet, GaussNet, MultiGaussNet, MultiClassNet, etc.),
similar to sklearn's make_regression, but with support for different response types
and signal-to-noise ratio (SNR) control. It also provides `make_x`, which
builds a design matrix from a data frame as R's `glmnet::makeX` does.
"""

import warnings

import numpy as np
import scipy.sparse
from scipy.special import expit, softmax
from numpy.random import default_rng
import pandas as pd
from statsmodels.genmod.families import Family as StatsmodelsFamily
from glmnet.glmnet import GLMNet  # Import at the top as requested


def make_dataset(estimator, n_samples=100, n_features=20, n_informative=10, n_targets=None, coef=None, snr=5, bias=0.0, random_state=None, **kwargs):
    """
    Generate a random regression, classification, or count dataset for GLMNet estimators or instances.

    This function can be used to generate synthetic data for regression, classification, or count models.
    It supports both estimator classes (e.g., LogNet, GaussNet) and GLMNet instances. If a GLMNet instance
    is provided, the function will use its family.base.rvs method (from statsmodels) to generate the response
    variable, allowing for custom family/link combinations.

    Parameters
    ----------
    estimator : type or GLMNet instance
        The GLMNet estimator class (e.g., LogNet, GaussNet, MultiGaussNet, MultiClassNet, FishNet),
        or an instance of GLMNet. If an instance is provided, its family.base.rvs method will be used
        to generate the response.
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The total number of features.
    n_informative : int, default=10
        The number of informative features.
    n_targets : int or None, default=None
        The dimension or the number of classes. Used only for MultiGaussNet and MultiClassNet. 
        For MultiClassNet it is interpreted as the number of classes.
    coef : array-like, default=None
        The coefficients to use. If None, random coefficients are generated.
    snr : float, default=5
        Desired signal-to-noise ratio. If set, noise will be scaled to achieve this SNR.
    bias : float, array-like, or None, default=0.0
        The bias (intercept) term in the underlying linear model. For multi-output, can be array-like.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray
        The output targets (regression, binary, multiclass, or count).
    coef : ndarray
        The underlying true coefficients used to generate the data.
    intercept : float or ndarray
        The intercept (bias) used in the data generation.

    Examples
    --------
    >>> from glmnet.paths import LogNet, GaussNet, MultiGaussNet, MultiClassNet
    >>> X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
    >>> X.shape, y.shape, coef.shape, np.shape(intercept)
    ((100, 10), (100,), (10,), ())
    >>> X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=100, n_features=10, n_targets=3, snr=5)
    >>> X.shape, y.shape, coef.shape, intercept.shape
    ((100, 10), (100, 3), (10, 3), (3,))
    >>> X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=100, n_features=10, n_targets=4, snr=2)
    >>> np.unique(y)
    array([0, 1, 2, 3])

    Using a GLMNet instance with a custom family:
    >>> from glmnet.glmnet import GLMNet
    >>> import statsmodels.api as sm
    >>> glmnet_instance = GLMNet(family=sm.families.Poisson())
    >>> X, y, coef, intercept = make_dataset(glmnet_instance, n_samples=100, n_features=10)
    >>> X.shape, y.shape, coef.shape, np.shape(intercept)
    ((100, 10), (100,), (10,), ())
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))

    # If estimator is an instance of GLMNet, use its family.base.rvs

    if not isinstance(estimator, type) and isinstance(estimator, GLMNet):

        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(size=n_informative)
        lin_pred = X @ coef + bias

        fam = estimator.family if not callable(estimator.family) else estimator.family()
        base = getattr(fam, 'base', None)
        mu = base.link.inverse(lin_pred)
        dist = base.get_distribution(mu, scale=1)
        y = dist.rvs(random_state=rng)
        intercept = bias
        return X, y, coef, intercept

    # Determine family/type from estimator class name
    name = estimator.__name__.lower()
    if 'multiclass' in name:
        family = 'multiclass'
    elif 'multigauss' in name:
        family = 'multigaussian'
    elif 'gauss' in name:
        family = 'gaussian'
    elif 'lognet' in name:
        family = 'binomial'
    elif 'fishnet' in name:
        family = 'poisson'
    else:
        raise ValueError(f"Unknown estimator class: {estimator}")

    if family == 'gaussian':
        # Standard regression
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept

        signal_var = np.var(lin_pred)
        noise_var = signal_var / snr
        noise = np.sqrt(noise_var)

        y = lin_pred + rng.normal(0, noise, size=n_samples)
    elif family == 'multigaussian':
        # Multi-output regression
        if n_targets is None:
            n_targets = 2
        if coef is None:
            coef = np.zeros((n_features, n_targets))
            for j in range(n_targets):
                coef[:n_informative, j] = rng.normal(0, 1, size=n_informative)
                rng.shuffle(coef[:, j])
        if bias is None:
            intercept = rng.normal(0, 1, size=n_targets)
        else:
            intercept = np.broadcast_to(bias, (n_targets,))
        lin_pred = X @ coef + intercept

        signal_var = np.var(lin_pred, axis=0)
        noise_var = signal_var / snr
        noise = np.sqrt(noise_var)

        y = lin_pred + rng.normal(0, noise, size=(n_samples, n_targets))
    elif family == 'binomial':
        # Binary classification
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept
        p = expit(lin_pred)
        if snr is not None:
            var_signal = np.var(lin_pred)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            lin_pred = lin_pred * scale
            p = expit(lin_pred)
        y = rng.binomial(1, p, size=n_samples)
    elif family == 'multiclass':
        # Multiclass classification
        if n_targets is None:
            n_targets = 3
        n_classes = n_targets
        if coef is None:
            coef = np.zeros((n_features, n_classes))
            for j in range(n_classes):
                coef[:n_informative, j] = rng.normal(0, 1, size=n_informative)
                rng.shuffle(coef[:, j])
        if bias is None:
            intercept = rng.normal(0, 1, size=n_classes)
        else:
            intercept = np.broadcast_to(bias, (n_classes,))
        lin_pred = X @ coef + intercept
        logits = lin_pred
        if snr is not None:
            var_signal = np.var(logits, axis=0)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            logits = logits * scale
        p = softmax(logits, axis=1)
        y = np.array([rng.choice(n_classes, p=p[i]) for i in range(n_samples)])
    elif family == 'poisson':
        # Poisson regression
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept
        mu = np.exp(lin_pred)
        if snr is not None:
            var_signal = np.var(lin_pred)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            lin_pred = lin_pred * scale
            mu = np.exp(lin_pred)
        y = rng.poisson(mu)
    else:
        raise ValueError(f"Unknown family: {family}.")
    return X, y, coef, intercept


def make_survival(n_samples=100, n_features=20, n_informative=10, coef=None, random_state=None, bias=None,
                  snr=None, baseline_hazard=0.1, start_id=False, discretize=False, **kwargs):
    """
    Generate a random survival (time-to-event) dataset for CoxNet models.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The total number of features.
    n_informative : int, default=10
        The number of informative features.
    coef : array-like, default=None
        The coefficients to use. If None, random coefficients are generated.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
    bias : float or None, default=None
        (Ignored; included for API compatibility.)
    snr : float or None, default=None
        Desired signal-to-noise ratio for the linear predictor. If set, the scale of the linear predictor is adjusted.
    baseline_hazard : float, default=0.1
        The baseline hazard rate for event time simulation.
    start_id : bool, default=False
        If True, include a 'start' column for (start, stop] survival data.
    discretize : bool, default=False
        If True, discretize times to 2 significant digits to create ties in the data.
        This is useful for testing tie-breaking methods in Cox regression.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : pandas.DataFrame
        The output DataFrame with columns: 'event', 'status', and optionally 'start'.
    coef : ndarray
        The underlying true coefficients used to generate the data.

    Examples
    --------
    >>> X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
    >>> y.columns
    Index(['start', 'event', 'status'], dtype='object')
    >>> X.shape, y.shape, coef.shape
    ((100, 10), (100, 3), (10,))
    
    >>> # Generate data with ties for testing tie-breaking methods
    >>> X, y, coef = make_survival(n_samples=100, n_features=10, discretize=True)
    >>> len(np.unique(y['event'])) < len(y['event'])  # Should have ties
    True
    """
    rng = default_rng(random_state)
    n_informative = min(n_informative, n_features)
    X = rng.standard_normal((n_samples, n_features))
    if coef is None:
        coef = np.zeros(n_features)
        coef[:n_informative] = rng.normal(0, 1, size=n_informative)
        rng.shuffle(coef)
    lin_pred = X @ coef
    if snr is not None:
        var_signal = np.var(lin_pred)
        var_noise = var_signal / snr
        scale = np.sqrt(var_signal / (var_signal + var_noise))
        lin_pred = lin_pred * scale
    # Simulate event times
    U = rng.uniform(0, 1, size=n_samples)
    duration = -np.log(U) / (baseline_hazard * np.exp(lin_pred))
    # Random censoring
    censor_time = rng.exponential(duration.mean(), size=n_samples)
    status = (duration <= censor_time).astype(int)
    observed_time = np.minimum(duration, censor_time)
    
    # Discretize times if requested
    if discretize:
        # Round to 2 significant digits to create ties
        observed_time = np.round(observed_time, decimals=1) + 0.2
        if start_id:
            # Also discretize start times
            start_times = np.zeros(n_samples)
            start_times = np.round(start_times, decimals=1)
    
    data = {'event': observed_time, 'status': status}
    if start_id:
        # Simulate start times that are strictly less than event times
        # Use a fraction of the event times to ensure they're reasonable
        start_times = observed_time * rng.uniform(0.1, 0.8, size=n_samples)
        # Ensure start times are strictly less than event times
        start_times = np.minimum(start_times, observed_time * 0.99)
        if discretize:
            start_times = np.round(start_times, decimals=1)
            # After discretization, ensure strict inequality
            start_times = np.maximum(np.minimum(start_times, observed_time - 0.1), 0)
        data = {'start': start_times, **data}
    y = pd.DataFrame(data)
    return X, y, coef 


def make_x(train, test=None, na_impute=False, sparse=False):
    """
    Build a design matrix from a data frame, as R's `glmnet::makeX`.

    Numeric columns are kept. Categorical, string and object columns are
    one-hot encoded with one column for every level (no level is dropped),
    named by the column name followed by the level, as `model.matrix` names
    them. Boolean columns become one column named with the suffix `TRUE`. A
    missing value in a categorical column makes all of its indicator columns
    missing.

    If `test` is given, `train` and `test` are encoded together, so both
    matrices have the same columns, including the levels seen in only one of
    them.

    Parameters
    ----------
    train: pd.DataFrame
        Training data.
    test: pd.DataFrame, optional
        Test data with the same columns as `train`.
    na_impute: bool
        Replace missing values in both matrices by the column means of the
        training matrix. The means are stored in `X.attrs['means']`.
    sparse: bool
        Return data frames with sparse columns. They can be passed to the
        C++ path estimators (`GaussNet`, `LogNet`, ...) directly.

    Returns
    -------
    X: pd.DataFrame
        The training design matrix; if `test` is given, the tuple
        `(X, X_test)`.

    Notes
    -----
    The levels of a categorical column are its categories, followed by any
    categories (or values) found only in `test`. The levels of a string or
    object column are its distinct values, sorted. R sorts them in the
    collation order of its locale, so the column order can differ from R's
    for strings that differ only in case; use a categorical column to fix
    the order.
    """
    if not isinstance(train, pd.DataFrame):
        raise TypeError('train must be a pandas DataFrame')
    if test is not None:
        if not isinstance(test, pd.DataFrame):
            raise TypeError('test must be a pandas DataFrame')
        missing = [c for c in train.columns if c not in test.columns]
        if missing:
            raise ValueError(f'test is missing columns {missing}')
    n_train = train.shape[0]
    n_test = 0 if test is None else test.shape[0]

    names, blocks, single_level = [], [], []
    for col in train.columns:
        x_tr = train[col]
        x_te = None if test is None else test[col]
        levels = _make_x_levels(col, x_tr, x_te)
        if levels is None:
            # numeric
            values = x_tr if x_te is None else pd.concat([x_tr, x_te], ignore_index=True)
            block = values.astype(float).to_numpy()[:, None]
            names.append(str(col))
            blocks.append(_make_x_block(block, sparse))
            continue
        values = x_tr.astype(object) if x_te is None else pd.concat([x_tr.astype(object),
                                                                     x_te.astype(object)],
                                                                    ignore_index=True)
        values = np.asarray(values, dtype=object)
        if levels == [True]:
            # boolean: a single column, as model.matrix codes a logical
            names.append(f'{col}TRUE')
        else:
            names.extend(f'{col}{level}' for level in levels)
            if len(levels) == 1:
                single_level.append(f'{col}{levels[0]}')
        blocks.append(_make_x_indicators(values, levels, sparse))

    if single_level:
        warnings.warn(f'Column(s) {", ".join(single_level)} are all 1, '
                      'due to factors with a single level')

    if sparse:
        X_all = (scipy.sparse.hstack(blocks, format='csc') if blocks else
                 scipy.sparse.csc_array((n_train + n_test, 0)))
        X_all = scipy.sparse.csc_array(X_all)
    else:
        X_all = np.hstack(blocks) if blocks else np.zeros((n_train + n_test, 0))

    means = None
    if na_impute:
        means = _make_x_means(X_all, n_train, sparse)
        if sparse:
            cols = np.repeat(np.arange(X_all.shape[1]), np.diff(X_all.indptr))
            nas = np.isnan(X_all.data)
            X_all.data[nas] = means[cols[nas]]
        else:
            rows, cols = np.nonzero(np.isnan(X_all))
            X_all[rows, cols] = means[cols]

    def _frame(rows, index):
        if sparse:
            # not DataFrame.sparse.from_spmatrix: its fill value may be nan
            block = scipy.sparse.csc_array(X_all[rows])
            df = pd.DataFrame({j: pd.arrays.SparseArray(block[:, [j]].toarray().ravel(), fill_value=0.)
                               for j in range(len(names))}, index=index)
            df.columns = names
        else:
            df = pd.DataFrame(X_all[rows], index=index, columns=names)
        if means is not None:
            df.attrs['means'] = pd.Series(means, index=names)
        return df

    X = _frame(np.arange(n_train), train.index)
    if test is None:
        return X
    return X, _frame(np.arange(n_train, n_train + n_test), test.index)


def _is_boolean(x):
    if pd.api.types.is_bool_dtype(x.dtype):
        return True
    if x.dtype == object:
        observed = x.dropna()
        return len(observed) > 0 and all(isinstance(v, (bool, np.bool_)) for v in observed)
    return False


def _make_x_levels(col, x_tr, x_te):
    """
    Levels of a column to one-hot encode, `[True]` for a boolean column, or
    None for a numeric column.
    """
    if isinstance(x_tr.dtype, pd.CategoricalDtype):
        levels = list(x_tr.cat.categories)
        if x_te is not None:
            new = (x_te.cat.categories if isinstance(x_te.dtype, pd.CategoricalDtype)
                   else sorted(x_te.dropna().unique()))
            levels.extend(v for v in new if v not in levels)
        return levels
    if _is_boolean(x_tr) and (x_te is None or _is_boolean(x_te) or x_te.isna().all()):
        return [True]
    if pd.api.types.is_numeric_dtype(x_tr.dtype):
        return None
    if pd.api.types.is_string_dtype(x_tr.dtype) or x_tr.dtype == object:
        observed = x_tr.dropna()
        if x_te is not None:
            observed = pd.concat([observed, x_te.dropna()])
        return sorted(observed.unique())
    raise TypeError(f'column {col!r} has unsupported dtype {x_tr.dtype}')


def _make_x_block(block, sparse):
    return scipy.sparse.csc_array(block) if sparse else block


def _make_x_indicators(values, levels, sparse):
    """
    Indicator columns of `levels`; all missing in rows where the value is
    missing. Values not in `levels` (False, for a boolean column) are 0.
    """
    n = len(values)
    isna = pd.isna(values)
    lookup = {level: j for j, level in enumerate(levels)}
    codes = np.array([-1 if na else lookup.get(v, -1) for v, na in zip(values, isna)], dtype=int)
    nas = np.nonzero(isna)[0]
    if sparse:
        observed = np.nonzero(codes >= 0)[0]
        rows = np.r_[observed, np.repeat(nas, len(levels))]
        cols = np.r_[codes[observed], np.tile(np.arange(len(levels)), len(nas))]
        data = np.r_[np.ones(len(observed)), np.full(len(nas) * len(levels), np.nan)]
        return scipy.sparse.csc_array((data, (rows, cols)), shape=(n, len(levels)))
    block = np.zeros((n, len(levels)))
    observed = codes >= 0
    block[np.nonzero(observed)[0], codes[observed]] = 1
    block[nas] = np.nan
    return block


def _make_x_means(X_all, n_train, sparse):
    """Column means of the training rows, ignoring missing values."""
    X_train = X_all[:n_train]
    if sparse:
        X_train = scipy.sparse.csc_array(X_train)
        nas = np.isnan(X_train.data)
        cols = np.repeat(np.arange(X_train.shape[1]), np.diff(X_train.indptr))
        n_na = np.bincount(cols[nas], minlength=X_train.shape[1])
        totals = np.bincount(cols[~nas], weights=X_train.data[~nas], minlength=X_train.shape[1])
        with np.errstate(invalid='ignore', divide='ignore'):
            return totals / (n_train - n_na)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(X_train, axis=0)
