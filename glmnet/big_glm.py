"""
Unpenalized fits with the glmnet machinery, as R's `glmnet::bigGlm`.
"""

import numpy as np
from sklearn.base import clone

from .glmnet import GLMNet, CoefPath


def big_glm(estimator, X, y, path=False):
    """
    Fit an unpenalized model (lambda = 0) with a glmnet estimator, as R's
    `glmnet::bigGlm`.

    All the other options of `estimator` are used: family, weights and
    offset (`weight_id`, `offset_id`), `exclude`, `lower_limits`,
    `upper_limits`, `fit_intercept`, `control`, ... . Penalty factors only
    matter through which variables are excluded (infinite factors).

    Parameters
    ----------
    estimator: GLMNet
        A glmnet estimator (`GaussNet`, `LogNet`, `FishNet`, `MultiGaussNet`,
        `MultiClassNet`, `CoxNet`, `GLMNet`, ...), or such a class. It is
        cloned, not modified.
    X: Union[np.ndarray, scipy.sparse, pd.DataFrame]
        Design matrix, as for `estimator.fit`.
    y: Union[np.ndarray, pd.DataFrame]
        Response, as for `estimator.fit`.
    path: bool
        If True, fit the default path of `estimator` first and then the path
        with lambda = 0 appended, so the unpenalized fit is reached by warm
        starts. This can be more stable, but takes longer.

    Returns
    -------
    estimator: GLMNet
        The fitted clone, with a single fit at lambda = 0: its `coefs_` and
        `intercepts_` have a first axis of length 1, and `predict` gives
        predictions from the unpenalized fit. Its `lambda_values` are set
        to `[0]`, so refitting it (or a clone) fits lambda = 0 alone.
    """
    if isinstance(estimator, type):
        estimator = estimator()
    estimator = clone(estimator)

    zero = np.array([0.])
    if not path:
        estimator.lambda_values = zero
        estimator.fit(X, y)
    else:
        lambda_values = clone(estimator).fit(X, y).lambda_values_
        estimator.lambda_values = np.r_[lambda_values, 0.]
        estimator.fit(X, y)
        if estimator.lambda_values_[-1] != 0:
            # GLMNet's IRLS path stops early once the deviance explained
            # stops changing: finish with lambda = 0 from where it stopped
            estimator.lambda_values = zero
            estimator.fit(X, y, warm_state=estimator.state_)
        _keep_last(estimator)
        estimator.lambda_values = zero

    _set_df(estimator)
    return estimator


def _keep_last(estimator):
    """Keep only the last fit on the path of a fitted estimator."""
    k = estimator.lambda_values_.shape[0] - 1
    fracdev = estimator.coef_path_.fracdev
    estimator.coefs_ = estimator.coefs_[k:]
    estimator.intercepts_ = estimator.intercepts_[k:]
    estimator.lambda_values_ = estimator.lambda_values_[k:]
    estimator.summary_ = estimator.summary_.iloc[k:]
    estimator.coef_path_ = CoefPath(coefs=estimator.coefs_,
                                    intercepts=estimator.intercepts_,
                                    lambda_values=estimator.lambda_values_,
                                    feature_names=estimator.feature_names_in_,
                                    fracdev=None if fracdev is None else np.asarray(fracdev)[k:])


def _set_df(estimator):
    # the path codes report 0 degrees of freedom at the first lambda
    coefs = estimator.coefs_
    nonzero = coefs != 0
    if coefs.ndim == 3:
        # multiple responses: a variable counts if any of its coefficients is nonzero
        nonzero = nonzero.any(axis=2)
    estimator.summary_ = estimator.summary_.copy()
    estimator.summary_['Degrees of Freedom'] = nonzero.sum(axis=1)
