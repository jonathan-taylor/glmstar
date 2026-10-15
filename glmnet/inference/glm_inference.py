"""
The inputs for selective inference after a glmnet / glmstar fit.

With the LASSO problem (beta_hat, G, Q_hat) on the rows used for selection
(:func:`glmnet.inference.glm_problem.glmnet_problem`; Q_hat the Fisher information at the
one-step relaxed fit by default), the selection score is Z_noisy = -G + Q_hat beta_hat, and
Var(Z_full) = dispersion * Q_hat / sum(weights).

Without randomization Z_full = Z_noisy (scalar_noise = 0, the polyhedral approach).
With held-out data, selecting on a proportion pi = sum(weights[rows]) / sum(weights) is
randomization with scalar_noise = (1 - pi) / pi, and Z_full linearizes the full-data
gradient at one Fisher scoring step for the full-data loss on the selected coordinates from
beta_hat,

    beta_tilde = beta_hat - I_EE^{-1} grad_E,   Z_full = -G_full(beta_tilde) + Q_hat beta_tilde.

Linearizing at the shrunk beta_hat instead biases Z_full and undercovers.

The selection can also be randomized without holding out data, by a linear term in the
objective (``GLMNet``'s ``linear_term``, a ``LinearTerm`` with an entry for the intercept):
the LASSO is fit to the data on all the rows with the linear term -omega, a perturbation of
the score that is independent of y given a pilot fit, with
Var(omega) = scalar_noise * Var(Z_full). :func:`score_perturbation` draws one from the
pilot fit. The response itself is never perturbed. Z_full is as for held-out data, from the
unrandomized score of (X, y).

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

from ..elnet import LinearTerm
from .glm_problem import (GLMProblem,
                          glm_moments,
                          resolve_family,
                          _variance,
                          _glmstar_family,
                          glmstar_problem)


@dataclass
class GLMInferenceProblem:
    """
    A GLMProblem fitted on the selection rows, with the full-data score
    ``Z_full``, its covariance ``Sigma`` and the randomization ``scalar_noise``
    (0 when the selection was not randomized).
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
    """Gradient of the unpenalized loss (1 / sum(w)) sum_i w_i (-l_i) at beta."""
    mu, a, _ = glm_moments(family, _eta(X, beta, offset, fit_intercept))
    r = weights / weights.sum() * a * (mu - y)
    g = X.T @ r
    return np.r_[r.sum(), g] if fit_intercept else g


def _newton_step(X, y, beta, family, weights, offset, fit_intercept):
    """
    One Fisher scoring step for the unpenalized loss on the nonzero coordinates of beta (and
    the intercept), with the information at beta.
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
    mu, a, W = glm_moments(family, _eta(X, beta, offset, fit_intercept))
    try:
        step = np.linalg.solve(X_active.T @ (X_active * (w * W)[:, None]),
                               X_active.T @ (w * a * (mu - y)))
    except np.linalg.LinAlgError:
        warnings.warn('the full-data information on the selected coordinates is singular; '
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


def score_perturbation(X,
                       eta,
                       family,
                       scalar_noise,
                       weights=None,
                       dispersion=1.,
                       fit_intercept=True,
                       method='replicates',
                       rng=None):
    """
    A randomization of the score for a randomized selection without held-out data: the
    ``LinearTerm`` -omega to pass to ``GLMNet(linear_term=...)``, fitted to the unperturbed data.

    With a pilot linear predictor ``eta`` (e.g. the unpenalized fit on all the variables when
    n > p) and its score weights a = h / V (h = d mu / d eta, V the variance function), draw
    e_i, independent of y given the pilot fit, with mean 0 and
    Var(e_i) = scalar_noise * dispersion * V(mu_i) / w_i. Then

        omega = (1 / sum(w)) [sum_i w_i a_i e_i,  sum_i w_i a_i e_i x_i]

    is the change in the score of y + e with the weights a frozen at the pilot fit, and
    Var(omega) = scalar_noise * dispersion * X1^T diag(w h^2 / V) X1 / sum(w)^2: scalar_noise
    times the covariance of the score. For the canonical link a = 1, and -omega is the linear
    term of the response y + e.

    Parameters
    ----------
    X : design (n x p), as passed to ``GLMNet``
    eta : pilot linear predictor (including any offset)
    family : as for :func:`glmnet.inference.glm_problem.glmnet_problem`
    scalar_noise : variance of omega relative to that of the score, e.g. (1 - pi) / pi to
        imitate selecting on a proportion pi of the data
    weights : prior weights, as passed to ``GLMNet``; default 1
    dispersion : multiplies the variance of y (e.g. a quasi-Poisson phi_hat)
    fit_intercept : whether the fit has an intercept (else omega has no intercept entry)
    method : 'replicates', e = sqrt(scalar_noise * dispersion) (y*_1 - y*_2) / sqrt(2) with
        y*_1, y*_2 independent replicates from the pilot fit (binomial and poisson weights
        must then be integers, the binomial ones numbers of trials); or 'gaussian', e normal
    rng : numpy Generator (or seed)

    Returns
    -------
    LinearTerm (-omega) on the original scale of the coefficients.
    """
    family = resolve_family(family)
    rng = np.random.default_rng(rng)
    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=float)
    eta = np.asarray(eta, dtype=float)
    n = X.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if scalar_noise <= 0:
        raise ValueError('scalar_noise must be positive')
    mu, a, _ = glm_moments(family, eta)
    if method == 'gaussian':
        V = _variance(family.name, mu)
        e = rng.standard_normal(n) * np.sqrt(scalar_noise * dispersion * V / weights)
    elif method == 'replicates':
        if family.name == 'gaussian':
            draws = [mu + rng.standard_normal(n) / np.sqrt(weights) for _ in range(2)]
        else:
            m = np.rint(weights)
            if not np.allclose(m, weights) or np.any(m < 1):
                raise ValueError("method='replicates' needs integer weights >= 1 for binomial and "
                                 "poisson; use method='gaussian'")
            m = m.astype(int)
            if family.name == 'binomial':
                draws = [rng.binomial(m, mu) / m for _ in range(2)]
            else:
                draws = [rng.poisson(m * mu) / m for _ in range(2)]
        e = np.sqrt(scalar_noise * dispersion) * (draws[0] - draws[1]) / np.sqrt(2)
    else:
        raise ValueError("method must be 'replicates' or 'gaussian'")
    r = weights / weights.sum() * a * e
    return LinearTerm(coef=-(X.T @ r), intercept=-r.sum() if fit_intercept else 0.)


def glm_inference_problem(problem,
                          X,
                          y,
                          family,
                          weights=None,
                          offset=None,
                          selection_rows=None,
                          dispersion=None,
                          scalar_noise=None,
                          Sigma=None):
    """
    The inputs for selective inference for the LASSO ``problem`` fitted on
    ``X[selection_rows]``, possibly with a linear term randomizing the selection.

    Parameters
    ----------
    problem : GLMProblem for the selection rows (from glmnet_problem or glmstar_problem),
        including the fit's linear term
    X, y : the full data (arrays)
    family : as for :func:`glmnet.inference.glm_problem.glmnet_problem`
    weights, offset : full-data observation weights and offset (prior weights, as in a GLM)
    selection_rows : rows (indices or boolean mask) used for selection; None for all of
        them
    dispersion : default 1 for binomial and poisson and, for gaussian with the identity
        link, the residual variance of the least squares fit on all the variables
        (requires n > p)
    scalar_noise : the variance of the randomization relative to Var(Z_full); default
        (1 - pi) / pi for selection on a proportion pi of the weights (0 for all the
        rows). Required when the problem has a linear term (e.g. from
        :func:`score_perturbation`): with held-out rows as well, it is the total.
    Sigma : Var(Z_full), the covariance of the score of the full data on the scale of G_hat
        (an array or LinearOperator); default dispersion * Q_hat / sum(weights), for
        independent observations from the model. Give it when they are dependent, e.g.
        X1^T Sigma_eps X1 / n^2 for gaussian errors with covariance Sigma_eps (unit weights,
        X1 = [1, X] with an intercept), or when the model is misspecified, e.g. the
        sandwich X1^T diag(w^2 a^2 (y - mu)^2) X1 / sum(w)^2, which as a LinearOperator
        (``XTVXOperator``) costs O(n p) per product. ``dispersion`` is then not used.

    Returns
    -------
    GLMInferenceProblem (index 0 of its vectors is the intercept when the problem has one).
    """
    family = resolve_family(family)
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
    lin = problem.linear_term
    if scalar_noise is None:
        if lin is not None:
            raise ValueError('the selection was randomized by a linear term: give its scalar_noise')
        pi_selection = weights[rows].sum() / weights.sum()
        scalar_noise = (1 - pi_selection) / pi_selection
    elif scalar_noise == 0 and (lin is not None or len(rows) < n):
        raise ValueError('a randomized selection (a linear term or held-out rows) has scalar_noise > 0')

    fit_intercept = problem.fit_intercept
    Q = problem.Q_hat
    b = problem.beta_hat
    if scalar_noise == 0:
        Z_full = -problem.G_hat + Q @ b
    else:
        b_tilde = _newton_step(X, y, b, family, weights, offset, fit_intercept)
        # G = loss gradient + ridge * beta + linear term; recover ridge * beta_hat
        ridge_b = problem.G_hat - _loss_gradient(X[rows], y[rows], b, family, weights[rows],
                                                 offset[rows], fit_intercept)
        if lin is not None:
            ridge_b = ridge_b - lin
        with np.errstate(divide='ignore', invalid='ignore'):
            ridge_b_tilde = np.where(b != 0, ridge_b * b_tilde / b, 0.)
        G_full = _loss_gradient(X, y, b_tilde, family, weights, offset, fit_intercept) + ridge_b_tilde
        Z_full = -G_full + Q @ b_tilde

    if Sigma is not None:
        return GLMInferenceProblem(problem=problem,
                                   Z_full=np.asarray(Z_full),
                                   Sigma=Sigma,
                                   scalar_noise=scalar_noise,
                                   dispersion=np.nan)
    if dispersion is None:
        if family.name == 'gaussian':
            if not family.canonical:
                raise ValueError('supply the dispersion for the gaussian family with a non-identity link')
            dispersion = _gaussian_dispersion(X, y, weights, offset, fit_intercept)
        else:
            dispersion = 1.
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
                              scalar_noise=None,
                              Sigma=None,
                              hessian='dense',
                              information='relaxed'):
    """
    The inputs for selective inference after a glmstar ``GLMNet`` fit at ``lambda_val``.

    Without ``selection_rows`` the fit is on all the data, randomized if it has a
    ``linear_term`` (give its ``scalar_noise``). With ``selection_rows``, ``glmnet_obj``
    must have been fitted on ``X[selection_rows]``, ``y[selection_rows]`` and Z_full uses
    all of ``X``, ``y`` (carving). ``y`` may be a DataFrame with response, weight and
    offset columns, as for glmstar. ``Sigma`` (an array or LinearOperator) overrides the
    covariance of the score, for dependent observations or a misspecified model. See :func:`glm_inference_problem`; ``hessian`` and
    ``information`` are as for :func:`glmnet.inference.glm_problem.glmnet_problem`.
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
                                 dispersion=dispersion,
                                 scalar_noise=scalar_noise,
                                 Sigma=Sigma)


__all__ = ['GLMInferenceProblem', 'glm_inference_problem', 'glmstar_inference_problem',
           'score_perturbation']
