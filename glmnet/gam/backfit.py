"""
Local scoring for GAMs: the state of a fit, and the regularizer whose Newton
step backfits the terms to the working response of IRLS (`glmnet.irls`).
"""
from dataclasses import dataclass, field
import logging
import warnings
from typing import Optional

import numpy as np
from statsmodels.genmod.families import family as sm_family

from glmnet.family import GLMState


@dataclass
class GAMState(object):
    """
    State of a GAM fit: the intercept and the fits of the terms at the
    training rows (`fits`, one column per term, each of mean 0).

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
    """
    fits: np.ndarray
    intercept: float
    obj_val: float = np.inf
    pmin: float = 1e-9

    def update(self,
               design,                    # ignored: the fits are at the training rows
               family,
               offset,
               objective=None):
        """Set the linear predictor, the mean and the objective, as `GLMState.update`."""
        base = family.base
        self.linear_predictor = self.intercept + self.fits.sum(1)
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = base.link.inverse(self.link_parameter)

        # shorthand
        self.mu = self.mean_parameter
        self.eta = self.linear_predictor

        if isinstance(base, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = base.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)

    logl_score = GLMState.logl_score


@dataclass
class GAMRegularizer(object):
    """
    The penalties of the terms of a GAM, and its Newton step: weighted
    backfitting of the terms to the working response.

    The objective is the deviance over twice the sum of the weights, plus the
    terms' penalties, plus the linear term `linear_term @ eta` (`eta` the
    linear predictor, without the offset). For the working problem of IRLS,
    with response `z` and weights `w`, the linear term moves the response to
    `z - linear_term / w`.

    Parameters
    ----------
    terms : list
        Terms of the GAM (`ISLP.gam.terms`), set up on the training data.
    linear_term : np.ndarray, optional
        Linear term in the linear predictor, one entry per observation.
    bf_maxit : int
        Maximum number of backfitting sweeps per Newton step.
    bf_tol : float
        Backfitting stops when the change in the fits, relative to
        `0.1 + |fits|` (Frobenius norms), is below `bf_tol`.
    """
    terms: list
    linear_term: Optional[np.ndarray] = None
    bf_maxit: int = 500
    bf_tol: float = 1e-10
    fit_intercept: bool = True
    warm_state: dict = field(default_factory=dict)

    @property
    def has_objective(self):
        return all(term.has_objective for term in self.terms)

    def half_step(self, state, oldstate):
        return GAMState(0.5 * (oldstate.fits + state.fits),
                        0.5 * (oldstate.intercept + state.intercept))

    def _debug_msg(self, state):
        return f'Intercept: {state.intercept}, Objective: {state.obj_val}'

    def check_state(self, state):
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
        w = np.asarray(sample_weight, float)
        z = np.asarray(pseudo_response, float)
        if self.linear_term is not None:
            omega = np.asarray(self.linear_term, float)
            if np.any((w <= 0) & (omega != 0)):
                raise ValueError('the linear term should be 0 where the weights are 0')
            z = z - np.divide(omega, w, out=np.zeros_like(omega), where=w > 0)

        for term in self.terms:
            term.set_weights(w)

        sw = w.sum()
        fits = np.array(cur_state.fits, float)
        total = fits.sum(1)
        intercept = (w * (z - total)).sum() / sw

        for sweep in range(self.bf_maxit):
            old = fits.copy()
            for j, term in enumerate(self.terms):
                partial = z - intercept - (total - fits[:, j])
                f = term.smooth(partial)
                f = f - f.mean()
                total += f - fits[:, j]
                fits[:, j] = f
                intercept = (w * (z - total)).sum() / sw
            change = np.linalg.norm(fits - old) / (0.1 + np.linalg.norm(fits))
            if change < self.bf_tol:
                break
        else:
            warnings.warn(f'GAM backfitting: the fits did not converge in {self.bf_maxit} '
                          f'sweeps (relative change {change:.2g}); increase bf_maxit')

        self.n_sweeps_ = sweep + 1
        self.warm_state = GAMState(fits, intercept)
        return self.warm_state

    def objective(self, state):
        value = sum(term.penalty(state.fits[:, j]) for j, term in enumerate(self.terms))
        if self.linear_term is not None:
            value += np.asarray(self.linear_term) @ state.linear_predictor
        return value
