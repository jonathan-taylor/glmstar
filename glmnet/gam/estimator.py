"""
Generalized additive models fitted by local scoring (Hastie & Tibshirani):
IRLS, whose working problems are solved by backfitting the terms.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from glmnet._utils import _get_data, _parent_dataclass_from_child
from glmnet.family import GLMFamilySpec
from glmnet.glm import GLM, GLMControl
from glmnet.irls import IRLS
from .backfit import GAMRegularizer, GAMState


@dataclass
class GAMControl(GLMControl):
    """
    Control parameters for fitting a GAM: those of `GLMControl` for the IRLS
    iterations, and for the backfitting of each working problem.

    Parameters
    ----------
    bf_maxit: int
        Maximum number of backfitting sweeps per IRLS iteration.
    bf_tol: float
        Backfitting stops when the change in the fits of the terms, relative
        to `0.1 + |fits|` (Frobenius norms), is below `bf_tol`.
    """
    bf_maxit: int = 500
    bf_tol: float = 1e-10


@dataclass
class GAM(BaseEstimator):
    """
    Generalized additive model, `g(E[y]) = intercept + sum_j f_j(x_j)`,
    fitted by local scoring: IRLS, each working problem solved by weighted
    backfitting of the terms (Hastie & Tibshirani, 1990).

    The terms (`ISLP.gam.s`, `lo`, `linear`, `factor`) each name a column
    of `X`. The smoothing of each term is fixed: by its smoothing parameter
    `lam`, or by its degrees of freedom `df` (as R's `gam`). When every term
    has a fixed smoothing parameter (`s(lam=...)`, `linear`, `factor`), the
    fit minimizes

        deviance / (2 sum(w)) + sum_j lam_j / 2 * int f_j''^2 + linear_term @ eta,

    with `eta` the linear predictor (without the offset) and `linear_term`
    the optional argument of `fit`. Otherwise (`df` terms, loess) local scoring
    stops on the change in `eta`.

    Parameters
    ----------
    terms: list
        Terms of the model.
    family: sm_family.Family
        Family (statsmodels) of the response. Default Gaussian.
    control: GAMControl
        Parameters for the IRLS iterations and the backfitting.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)

    Attributes
    ----------
    terms_: list
        The fitted terms (copies of `terms`).
    intercept_: float
        Intercept: the fits of the terms have mean 0 over the training rows.
    fits_: pd.DataFrame or np.ndarray
        The fits of the terms at the training rows, one column per term.
    edf_: pd.Series
        Degrees of freedom of each term, without the constant: for a spline,
        the trace of its smoother at the final weights less 1.
    df_resid_: float
        Residual degrees of freedom, the number of observations (of positive
        weight) less 1 less the sum of `edf_` (as R's `gam`).
    deviance_, null_deviance_, dispersion_, loglike_, aic_: float
        As for `GLM`; the AIC counts `1 + sum(edf_)` parameters.
    n_iter_: int
        Number of IRLS iterations.
    """
    terms: list = field(default_factory=list)
    family: object = field(default_factory=GLMFamilySpec)
    control: GAMControl = field(default_factory=GAMControl)
    offset_id: Union[str, int] = None
    weight_id: Union[str, int] = None
    response_id: Union[str, int] = None

    fit_intercept = True

    def _columns(self, X):
        """The columns of `X` of the terms."""
        cols = []
        for term in self.terms:
            if isinstance(X, pd.DataFrame):
                if isinstance(term.column, str):
                    cols.append(X[term.column].to_numpy())
                else:
                    cols.append(X.iloc[:, term.column].to_numpy())
            else:
                cols.append(np.asarray(X)[:, term.column])
        return cols

    def fit(self,
            X,
            y,
            linear_term=None,
            dispersion=None):
        """
        Fit the GAM.

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            The data: the terms name its columns.
        y: Union[np.ndarray, pd.DataFrame]
            Response (with offset and weights, see `offset_id`, `weight_id`).
        linear_term: np.ndarray, optional
            A linear term `linear_term @ eta` added to the objective, one entry
            per observation (0 where the weight is 0), with `eta` the linear
            predictor without the offset; as `GLMNet`'s `linear_term`, on the
            scale of the deviance over twice the sum of the weights.
        dispersion: float, optional
            Dispersion. By default it is estimated, by Pearson's chi-squared
            over `df_resid_`, for families with a free dispersion, and is 1
            for the binomial and Poisson.

        Returns
        -------
        self
        """
        if not self.terms:
            raise ValueError('a GAM needs at least one term')
        if self.control is None:
            self.control = GAMControl()
        elif isinstance(self.control, dict):
            self.control = _parent_dataclass_from_child(GAMControl, self.control)

        self._train_index = X.index if isinstance(X, pd.DataFrame) else None
        nobs = X.shape[0]

        family = self._family = GLMFamilySpec.from_family(self.family, y)
        _, _, response, offset, weight = _get_data(self,
                                                   np.zeros((nobs, 1)),
                                                   y,
                                                   offset_id=self.offset_id,
                                                   response_id=self.response_id,
                                                   weight_id=self.weight_id,
                                                   check=False)
        response = np.asarray(response, float).reshape(-1)
        weight = np.asarray(weight, float).reshape(-1)
        self._response, self._weight = response, weight
        weight_total = weight.sum()
        normed_weight = weight / weight_total
        self.weight_total_ = weight_total

        self.terms_ = [deepcopy(term).setup(x) for term, x in zip(self.terms,
                                                                 self._columns(X))]
        if linear_term is not None:
            linear_term = np.asarray(linear_term, float).reshape(-1)
            if linear_term.shape != (nobs,):
                raise ValueError(f'linear_term should have shape {(nobs,)}')

        regularizer = self.regularizer_ = GAMRegularizer(terms=self.terms_,
                                                         linear_term=linear_term,
                                                         bf_maxit=self.control.bf_maxit,
                                                         bf_tol=self.control.bf_tol)

        null_state, null_deviance = family.get_null_deviance(response=response,
                                                             sample_weight=normed_weight,
                                                             offset=offset,
                                                             fit_intercept=True)
        self.null_deviance_ = weight_total * null_deviance

        def obj_function(state):
            return (family.deviance(response, state.mu, normed_weight) / 2 +
                    regularizer.objective(state))

        state = GAMState(np.zeros((nobs, len(self.terms_))),
                         float(null_state.coef[0]))
        state.update(None, family, offset, obj_function)

        info = {}
        converged, boundary, state, _ = IRLS(regularizer,
                                             family,
                                             None,
                                             response,
                                             offset,
                                             normed_weight,
                                             state,
                                             obj_function,
                                             self.control,
                                             info=info)
        self.n_iter_ = info['n_iter']
        if not converged:
            import warnings
            warnings.warn(f'GAM: local scoring did not converge in mxitnr={self.control.mxitnr} '
                          'iterations; increase control.mxitnr')

        # the terms at the weights of the final working problem
        z, w = family.get_response_and_weights(state, response, offset, normed_weight)
        total = state.fits.sum(1)
        coefs = state.coefs if state.coefs is not None else [None] * len(self.terms_)
        for j, term in enumerate(self.terms_):
            term.set_weights(w)
            term.finalize(state.fits[:, j], z - state.intercept - (total - state.fits[:, j]),
                          coefs[j])

        names = [term.name for term in self.terms_]
        self.state_ = state
        self.intercept_ = state.intercept
        self.fits_ = self._on_train_index_frame(state.fits, names)
        self.edf_ = pd.Series([term.edf() for term in self.terms_], index=names, name='edf')

        self.deviance_ = weight_total * family.deviance(response, state.mu, normed_weight)
        self.df_resid_ = float(np.sum(weight > 0) - 1 - self.edf_.sum())

        self._estimated_dispersion = dispersion is None and not self._fixed_dispersion()
        if self._estimated_dispersion:
            dispersion = (np.sum(self.residuals('pearson')**2) / self.df_resid_
                          if self.df_resid_ > 0 else np.nan)
        elif dispersion is None:
            dispersion = 1.
        self.dispersion_ = dispersion

        k, n = 1 + self.edf_.sum(), np.sum(weight > 0)
        self.loglike_ = self._loglike()
        self.aic_ = -2 * self.loglike_ + 2 * k
        self.bic_ = -2 * self.loglike_ + np.log(n) * k
        return self

    def _on_train_index_frame(self, value, names):
        index = getattr(self, '_train_index', None)
        if index is None:
            return value
        return pd.DataFrame(value, index=index, columns=names)

    def partial_effect(self, term, x):
        """
        The fitted function of a term at `x`: `term` is its index in `terms`
        or its name (e.g. `'s(age)'`).
        """
        if not isinstance(term, (int, np.integer)):
            term = [t.name for t in self.terms_].index(term)
        return self.terms_[term].predict(x)

    def _linear_predictor(self, X, offset=None):
        eta = self.intercept_ + sum(term.predict(x) for term, x in zip(self.terms_,
                                                                       self._columns(X)))
        if offset is not None:
            eta = eta + offset
        return np.asarray(eta, float)

    def predict(self, X, prediction_type='response', offset=None):
        """
        Predictions at `X`: of the mean (`'response'`) or of the linear
        predictor (`'link'`).
        """
        eta = self._linear_predictor(X, offset)
        if prediction_type == 'link':
            return eta
        elif prediction_type == 'response':
            return self._family.base.link.inverse(eta)
        raise ValueError("prediction_type should be one of 'response' or 'link'")

    def summary(self):
        """The terms, their degrees of freedom and smoothing parameters."""
        lam = [getattr(term, 'lam_', np.nan) for term in self.terms_]
        return pd.DataFrame({'edf': self.edf_.values, 'lam': lam},
                            index=self.edf_.index)

    # as for GLM: they use only the response, weights, family and state_
    residuals = GLM.residuals
    _on_train_index = GLM._on_train_index
    _fixed_dispersion = GLM._fixed_dispersion
    _loglike = GLM._loglike
