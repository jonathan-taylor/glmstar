"""
Local scoring for GAMs: the state of a fit, and the regularizer whose Newton
step backfits the terms to the working response of IRLS (`glmnet.irls`).

References
----------
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
  additive models. *The Annals of Statistics* 17(2), 453–510.
"""
from dataclasses import dataclass, field
import logging
import warnings
from typing import Optional

import numpy as np
from statsmodels.genmod.families import family as sm_family

from glmnet.family import GLMState
from glmnet.cox import CoxState
from .terms import _weighted_fit


@dataclass
class GAMState(object):
    """
    State of a GAM fit: the intercept and the fits of the terms at the
    training rows (`fits`, one column per term, each of mean 0 but for the
    terms that are not centered, random effects).

    Parameters
    ----------
    fits : np.ndarray
        Fits of the terms, of shape `(nobs, nterms)`.
    intercept : float
        Intercept.
    obj_val : float, default=np.inf
        Objective function value.
    pmin : float, default=1e-9
        Minimum probability for binomial family.
    coefs : list, optional
        Coefficients of the terms (`Smoother.coef()`, linear in their fits), one
        entry per term; None for a term without them, or whose fit is 0.
    """
    fits: np.ndarray
    intercept: float
    obj_val: float = np.inf
    pmin: float = 1e-9
    coefs: Optional[list] = None

    def update(self,
               design,                    # ignored: the fits are at the training rows
               family,
               offset,
               objective=None):
        """
        Set the linear predictor, the mean and the objective, as
        `GLMState.update` (`CoxState.update` for the Cox family, which has no
        mean: it is the linear predictor).
        """
        base = getattr(family, 'base', None)
        self.linear_predictor = self.intercept + self.fits.sum(1)
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = (self.link_parameter if base is None
                               else base.link.inverse(self.link_parameter))

        # shorthand
        self.mu = self.mean_parameter
        self.eta = self.linear_predictor

        if isinstance(base, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = base.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)

    def logl_score(self, family, y, sample_weight):
        """The score of the log-likelihood in the linear predictor (GLM's or Cox's)."""
        if getattr(family, 'base', None) is None:
            return CoxState.logl_score(self, family, y, sample_weight)
        return GLMState.logl_score(self, family, y, sample_weight)


@dataclass
class GAMRegularizer(object):
    """
    The penalties of the terms of a GAM, and its Newton step: weighted
    backfitting of the terms to the working response.

    With `modified=True` (the default, as R's `gam`) backfitting is modified
    (Buja, Hastie & Tibshirani, 1989): each sweep fits the constant and the
    terms' linear columns (the linear terms, the factors' indicators and the
    linear parts of the splines) jointly, by weighted least squares, then
    backfits only the splines' parts beyond them (and loess terms whole). It
    has the same fixed point as plain backfitting, as the splines are
    symmetric smoothers; the linear parts are not left to converge term by
    term, which is slow, or never, when the predictors are (nearly) collinear.

    The objective is the deviance over twice the sum of the weights, plus the
    terms' penalties, plus the linear term `linear_term @ eta` (`eta` the
    linear predictor, without the offset). For the working problem of IRLS,
    with response `z` and weights `w`, the linear term moves the response to
    `z - linear_term / w`.

    Parameters
    ----------
    terms : list
        Terms of the GAM (`glmnet.gam.terms`), set up on the training data.
    linear_term : np.ndarray, optional
        Linear term in the linear predictor, one entry per observation.
    bf_maxit : int
        Maximum number of backfitting sweeps per Newton step.
    bf_tol : float
        Backfitting stops when the change in the fits, relative to
        `0.1 + |fits|` (Frobenius norms), is below `bf_tol`.
    modified : bool
        Modified backfitting (the default) or plain backfitting.
    """
    terms: list
    linear_term: Optional[np.ndarray] = None
    bf_maxit: int = 500
    bf_tol: float = 1e-10
    modified: bool = True
    fit_intercept: bool = True
    warm_state: dict = field(default_factory=dict)
    # fit_intercept=False: for a family whose deviance does not change with
    # a constant added to the linear predictor (Cox), the constant of each
    # step is dropped (backfitting still fits it: the terms are centered)

    @property
    def has_objective(self):
        """Whether every term minimizes a fixed objective."""
        return all(term.has_objective for term in self.terms)

    def half_step(self, state, oldstate):
        """The state halfway between `oldstate` and `state` (fits and coefficients)."""
        return GAMState(0.5 * (oldstate.fits + state.fits),
                        0.5 * (oldstate.intercept + state.intercept),
                        coefs=[_average(c_old, c) for c_old, c in zip(_coefs(oldstate, self.terms),
                                                                      _coefs(state, self.terms))])

    def _debug_msg(self, state):
        return f'Intercept: {state.intercept}, Objective: {state.obj_val}'

    def check_state(self, state):
        """Raise if the fits or the intercept have NaNs."""
        if np.any(np.isnan(state.fits)):
            raise ValueError('fits have NaNs')
        if np.isnan(state.intercept):
            raise ValueError('intercept is NaN')

    def newton_step(self,
                    design,                 # ignored
                    pseudo_response,
                    sample_weight,
                    cur_state):
        """
        Backfit the terms to the working response `pseudo_response` with
        weights `sample_weight`, from the fits of `cur_state`, until the fits
        change by less than `bf_tol` (relatively), or for `bf_maxit` sweeps.
        """
        z, w = self._working(pseudo_response, sample_weight)

        fits = np.array(cur_state.fits, float)
        coefs = list(_coefs(cur_state, self.terms))
        if self.modified:
            fits, intercept, coefs, sweep, change = self._modified_backfit(z, w, fits, coefs)
        else:
            fits, intercept, coefs, sweep, change = self._backfit(z, w, fits, coefs)
        if change >= self.bf_tol:
            warnings.warn(f'GAM backfitting: the fits did not converge in {self.bf_maxit} '
                          f'sweeps (relative change {change:.2g}); increase bf_maxit')

        self.n_sweeps_ = sweep + 1
        if not self.fit_intercept:
            intercept = 0.
        self.warm_state = GAMState(fits, intercept, coefs=coefs)
        return self.warm_state

    def _working(self, pseudo_response, sample_weight):
        """
        The working response, moved by the linear term, and the weights; the
        terms are given the weights.
        """
        # (rounding can leave a weight just below 0: the diagonal of the
        # Hessian of Efron's partial likelihood with start times)
        w = np.maximum(np.asarray(sample_weight, float), 0)
        z = np.asarray(pseudo_response, float)
        if self.linear_term is not None:
            omega = np.asarray(self.linear_term, float)
            if np.any((w <= 0) & (omega != 0)):
                raise ValueError('the linear term should be 0 where the weights are 0')
            z = z - np.divide(omega, w, out=np.zeros_like(omega), where=w > 0)
        for term in self.terms:
            term.set_weights(w)
        return z, w

    def _backfit(self, z, w, fits, coefs):
        """Plain (Gauss-Seidel) backfitting."""
        sw = w.sum()
        total = fits.sum(1)
        intercept = (w * (z - total)).sum() / sw

        for sweep in range(self.bf_maxit):
            old = fits.copy()
            for j, term in enumerate(self.terms):
                partial = z - intercept - (total - fits[:, j])
                f = term.smooth(partial)
                center = f.mean() if term.centered else 0.
                f = f - center
                coef = term.coef()
                coefs[j] = None if coef is None else coef - center
                total += f - fits[:, j]
                fits[:, j] = f
                intercept = (w * (z - total)).sum() / sw
            change = np.linalg.norm(fits - old) / (0.1 + np.linalg.norm(fits))
            if change < self.bf_tol:
                break
        return fits, intercept, coefs, sweep, change

    def _modified_backfit(self, z, w, fits, coefs):
        """
        Modified backfitting: the constant and the terms' linear columns
        jointly by weighted least squares, then the smooths' other parts.
        """
        n, terms = z.shape[0], self.terms
        columns, slices, start = [np.ones((n, 1))], [], 1
        for term in terms:
            X_j = term.linear_columns()
            if X_j is None:
                slices.append(None)
            else:
                columns.append(X_j)
                slices.append(slice(start, start + X_j.shape[1]))
                start += X_j.shape[1]
        X = np.hstack(columns)

        # the weighted least squares projection on the columns of X (of
        # minimum norm if they are collinear)
        sq = np.sqrt(w)
        U, d, Vt = np.linalg.svd(sq[:, None] * X, full_matrices=False)
        keep = d > d[0] * 1e-10
        U, d, Vt = U[:, keep], d[keep], Vt[keep]

        def project(r):
            return Vt.T @ ((U.T @ (sq * r)) / d)

        smooths = [j for j, term in enumerate(terms) if term.nonlinear]
        # the parts beyond the linear columns, from the current fits
        nonlin = np.zeros_like(fits)
        for j in smooths:
            if not terms[j].centered:
                # its smoother reproduces no function: all of it is backfitted
                nonlin[:, j] = fits[:, j]
                continue
            X_j = None if slices[j] is None else X[:, slices[j]]
            nonlin[:, j] = fits[:, j] - _weighted_fit(X_j, fits[:, j], w)
        nonlin_total = nonlin.sum(1)
        beta = project(z - nonlin_total)

        for sweep in range(self.bf_maxit):
            old = fits.copy()
            linear = X @ beta
            for j in smooths:
                partial = z - linear - (nonlin_total - nonlin[:, j])
                f = terms[j].smooth_nonlinear(partial)
                nonlin_total += f - nonlin[:, j]
                nonlin[:, j] = f
            beta = project(z - nonlin_total)

            intercept = beta[0]
            for j, term in enumerate(terms):
                f = nonlin[:, j].copy()
                beta_j = None if slices[j] is None else beta[slices[j]]
                if beta_j is not None:
                    f += X[:, slices[j]] @ beta_j
                center = f.mean() if term.centered else 0.
                fits[:, j] = f - center
                intercept += center
                coefs[j] = term.combine_coef(beta_j if beta_j is not None else np.zeros(0),
                                             center)
            change = np.linalg.norm(fits - old) / (0.1 + np.linalg.norm(fits))
            if change < self.bf_tol:
                break
        return fits, intercept, coefs, sweep, change

    def objective(self, state):
        """The terms' penalties at `state`, plus the linear term."""
        coefs = _coefs(state, self.terms)
        value = sum(term.penalty(state.fits[:, j], coef)
                    for j, (term, coef) in enumerate(zip(self.terms, coefs)))
        if self.linear_term is not None:
            value += np.asarray(self.linear_term) @ state.linear_predictor
        return value


def _coefs(state, terms):
    return state.coefs if state.coefs is not None else [None] * len(terms)


def _average(a, b):
    # None stands for the coefficients of the zero function
    if a is None and b is None:
        return None
    a = np.zeros_like(b) if a is None else a
    b = np.zeros_like(a) if b is None else b
    return 0.5 * (a + b)
