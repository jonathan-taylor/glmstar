"""
The inputs for selective inference after a glmnet / glmstar fit.

With the LASSO problem (beta_hat, G, Q_hat) on the rows used for selection
(:func:`glmnet.inference.glm_problem.glmnet_problem`; Q_hat at the one-step relaxed fit by
default), the selection score is Z_noisy = -G + Q_hat beta_hat, and
Var(Z_full) = dispersion * Q_hat / sum(weights).

Without held-out data Z_full = Z_noisy (scalar_noise = 0, the polyhedral approach).
With held-out data, selecting on a proportion pi = sum(weights[rows]) / sum(weights) is
randomization with scalar_noise = (1 - pi) / pi, and Z_full linearizes the full-data
gradient at one Newton step for the full-data loss on the selected coordinates from
beta_hat,

    beta_tilde = beta_hat - H_EE^{-1} grad_E,   Z_full = -G_full(beta_tilde) + Q_hat beta_tilde.

Linearizing at the shrunk beta_hat instead biases Z_full and undercovers.

The inference itself is done elsewhere, e.g. with ``lassoinf``::

    info = glmstar_inference_problem(G, X, y, lambda_val=lam)
    LassoInference(**info.lasso_args(), level=0.95)

Adapted from ``lassoinf/glm_inference.py`` (github.com/jonathan-taylor/lassoinf).
"""

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .glm_problem import (GLMProblem,
                          _mean_and_variance,
                          _glmstar_family,
                          glmstar_problem,
                          FAMILIES)


@dataclass
class GLMInferenceProblem:
    """
    A GLMProblem fitted on the selection rows, with the full-data score
    ``Z_full``, its covariance ``Sigma`` and the randomization ``scalar_noise``
    (0 when all the rows were used for selection).
    """
    problem: GLMProblem
    Z_full: np.ndarray
    Sigma: object
    scalar_noise: float
    dispersion: float

    def lasso_args(self):
        """Keyword arguments for lassoinf.LassoInference (add level, ...)."""
        return dict(**self.problem.lasso_args(),
                    Z_full=self.Z_full,
                    Sigma=self.Sigma,
                    Sigma_noise=None,
                    scalar_noise=self.scalar_noise)


def _eta(X, beta, offset, fit_intercept):
    if fit_intercept:
        return offset + beta[0] + X @ beta[1:]
    return offset + X @ beta


def _loss_gradient(X, y, beta, family, weights, offset, fit_intercept):
    """Gradient of the unpenalized loss (1 / sum(w)) sum_i w_i l_i at beta."""
    mu, _ = _mean_and_variance(family, _eta(X, beta, offset, fit_intercept))
    r = weights / weights.sum() * (mu - y)
    g = X.T @ r
    return np.r_[r.sum(), g] if fit_intercept else g


def _newton_step(X, y, beta, family, weights, offset, fit_intercept):
    """
    One Newton step for the unpenalized loss on the nonzero coordinates of beta (and the
    intercept), with the Hessian at beta.
    """
    active = np.nonzero(beta != 0)[0]
    if fit_intercept:
        active = np.union1d([0], active)
    if len(active) == 0:
        return beta
    cols = active[active > 0] - 1 if fit_intercept else active
    X_active = X[:, cols]
    if fit_intercept:
        X_active = np.column_stack([np.ones(X.shape[0]), X_active])
    w = weights / weights.sum()
    mu, var = _mean_and_variance(family, _eta(X, beta, offset, fit_intercept))
    try:
        step = np.linalg.solve(X_active.T @ (X_active * (w * var)[:, None]), X_active.T @ (w * (mu - y)))
    except np.linalg.LinAlgError:
        warnings.warn('the full-data Hessian on the selected coordinates is singular; '
                      'linearizing at the LASSO solution')
        return beta
    beta = beta.copy()
    beta[active] -= step
    return beta


def _gaussian_dispersion(X, y, weights, offset, fit_intercept):
    """Residual variance of the weighted least squares fit on all the variables."""
    X1 = np.column_stack([np.ones(X.shape[0]), X]) if fit_intercept else X
    n, k = X1.shape
    if n <= k:
        raise ValueError('cannot estimate the dispersion with n <= p (+ intercept); supply dispersion')
    sw = np.sqrt(weights)
    coef, _, rank, _ = np.linalg.lstsq(X1 * sw[:, None], (y - offset) * sw, rcond=None)
    resid = y - offset - X1 @ coef
    return np.sum(weights * resid**2) / (n - rank)


def _scale(Q, c):
    if isinstance(Q, LinearOperator):
        return aslinearoperator(Q) * c
    return c * Q


def glm_inference_problem(problem,
                          X,
                          y,
                          family,
                          weights=None,
                          offset=None,
                          selection_rows=None,
                          dispersion=None):
    """
    The inputs for selective inference for the LASSO ``problem`` fitted on
    ``X[selection_rows]``.

    Parameters
    ----------
    problem : GLMProblem for the selection rows (from glmnet_problem or glmstar_problem)
    X, y : the full data (arrays)
    family : 'gaussian', 'binomial' or 'poisson'
    weights, offset : full-data observation weights and offset (prior weights, as in a GLM)
    selection_rows : rows (indices or boolean mask) used for selection; None for all of
        them, i.e. no randomization
    dispersion : default 1 for binomial and poisson and, for gaussian, the residual
        variance of the least squares fit on all the variables (requires n > p)

    Returns
    -------
    GLMInferenceProblem (index 0 of its vectors is the intercept when the problem has one).
    """
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {FAMILIES}, got {family!r}')
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    if y.shape[0] != n or weights.shape[0] != n or offset.shape[0] != n:
        raise ValueError('y, weights and offset must be for the full data (length n)')
    rows = np.arange(n) if selection_rows is None else np.arange(n)[selection_rows]
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError('invalid selection_rows')

    fit_intercept = problem.fit_intercept
    Q = problem.Q_hat
    b = problem.beta_hat
    if len(rows) == n:
        Z_full = -problem.G_hat + Q @ b
        scalar_noise = 0.
    else:
        b_tilde = _newton_step(X, y, b, family, weights, offset, fit_intercept)
        # G = loss gradient + ridge * beta; recover ridge * beta_hat from the selection problem
        ridge_b = problem.G_hat - _loss_gradient(X[rows], y[rows], b, family, weights[rows],
                                                 offset[rows], fit_intercept)
        with np.errstate(divide='ignore', invalid='ignore'):
            ridge_b_tilde = np.where(b != 0, ridge_b * b_tilde / b, 0.)
        G_full = _loss_gradient(X, y, b_tilde, family, weights, offset, fit_intercept) + ridge_b_tilde
        Z_full = -G_full + Q @ b_tilde
        pi_selection = weights[rows].sum() / weights.sum()
        scalar_noise = (1 - pi_selection) / pi_selection

    if dispersion is None:
        dispersion = (_gaussian_dispersion(X, y, weights, offset, fit_intercept)
                      if family == 'gaussian' else 1.)
    return GLMInferenceProblem(problem=problem,
                               Z_full=np.asarray(Z_full),
                               Sigma=_scale(Q, dispersion / weights.sum()),
                               scalar_noise=scalar_noise,
                               dispersion=dispersion)


def _take(obj, rows):
    return obj.iloc[rows] if hasattr(obj, 'iloc') else obj[rows]


def glmstar_inference_problem(glmnet_obj,
                              X,
                              y,
                              lambda_val=None,
                              selection_rows=None,
                              dispersion=None,
                              hessian='dense',
                              information='relaxed'):
    """
    The inputs for selective inference after a glmstar ``GLMNet`` fit at ``lambda_val``.

    Without ``selection_rows`` the fit is on all the data and there is no randomization.
    With ``selection_rows``, ``glmnet_obj`` must have been fitted on
    ``X[selection_rows]``, ``y[selection_rows]`` and Z_full uses all of ``X``, ``y``
    (carving). ``y`` may be a DataFrame with response, weight and offset columns, as for
    glmstar. See :func:`glm_inference_problem`; ``hessian`` and ``information`` are as for
    :func:`glmnet.inference.glm_problem.glmnet_problem`.
    """
    n = X.shape[0]
    rows = np.arange(n) if selection_rows is None else np.arange(n)[selection_rows]
    problem = glmstar_problem(glmnet_obj, _take(X, rows), _take(y, rows), lambda_val=lambda_val,
                              hessian=hessian, information=information)
    X_arr, _, response, offset, weight = glmnet_obj.get_data_arrays(X, y)
    X_arr = np.asarray(X_arr.toarray() if sp.issparse(X_arr) else X_arr, dtype=float)
    return glm_inference_problem(problem, X_arr, response, _glmstar_family(glmnet_obj),
                                 weights=weight, offset=offset,
                                 selection_rows=None if len(rows) == n else rows,
                                 dispersion=dispersion)


__all__ = ['GLMInferenceProblem', 'glm_inference_problem', 'glmstar_inference_problem']
