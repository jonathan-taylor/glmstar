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
from statsmodels.genmod.families import family as sm_family

from glmnet._utils import _get_data, _parent_dataclass_from_child
from glmnet.family import GLMFamilySpec
from glmnet.cox import CoxFamily, CoxFamilySpec
from glmnet.glm import GLM, GLMControl
from glmnet.irls import IRLS
from .backfit import GAMRegularizer, GAMState
from .terms import Term


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
    bf_modified: bool
        Modified backfitting (the default, as R's `gam`): each sweep fits the
        linear parts of all the terms jointly, then backfits the smooths'
        nonlinear parts. False: plain backfitting.
    """
    bf_maxit: int = 500
    bf_tol: float = 1e-10
    bf_modified: bool = True


@dataclass
class GAM(BaseEstimator):
    """
    Generalized additive model, `g(E[y]) = intercept + sum_j f_j(x_j)`,
    fitted by local scoring: IRLS, each working problem solved by weighted
    backfitting of the terms (Hastie & Tibshirani, 1990).

    The model is a list of terms (`glmnet.gam.s`, `lo`, `linear`, `factor`,
    or a `ColumnsTerm` of several columns fitted as one parametric block),
    each naming a column of `X` (ISLP's `ISLP.gam` builds them from a
    ModelSpec).
    The smoothing of each term is fixed: by its smoothing parameter
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
    family: sm_family.Family or glmnet.cox.CoxFamily
        Family (statsmodels) of the response. Default Gaussian. Or the Cox
        model's (glmstar's `CoxFamily`, with Efron or Breslow ties, start
        times and strata): `y` is then a DataFrame of the event data (and
        the offset and weights), the fit has no intercept and maximizes the
        penalized partial likelihood, with the diagonal of its Hessian as
        the working weights; `predict` gives the linear predictor (`'link'`)
        or the relative risk `exp(eta)` (`'response'`).
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
        As for `GLM`; the AIC counts `1 + sum(edf_)` parameters, plus 1 for a
        free dispersion, as R's `gam` (for the Gamma and inverse Gaussian the
        log-likelihood is at the dispersion deviance / sum(weights), as R's).
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

    # hooks for a subclass that builds the terms and the columns they name
    # from a model specification (ISLP's GAM, from a ModelSpec)

    def _model_terms(self, X):
        """The terms of the model and the data whose columns they name."""
        terms = list(self.terms)
        for term in terms:
            if not isinstance(term, Term):
                raise ValueError(f'{term!r} is not a GAM term (s(), lo(), linear(), factor())')
        return terms, X

    def _design(self, X):
        """The data whose columns the fitted terms name (`X`)."""
        return X

    @staticmethod
    def _columns(X, terms):
        """The columns of `X` of the terms."""
        cols = []
        for term in terms:
            if isinstance(X, pd.DataFrame):
                # a name or a list of names (a ColumnsTerm), else positions
                labels = term.column if isinstance(term.column, list) else [term.column]
                if all(isinstance(c, str) for c in labels):
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
        terms, X = self._model_terms(X)
        if not terms:
            raise ValueError('a GAM needs at least one term')
        if self.control is None:
            self.control = GAMControl()
        elif isinstance(self.control, dict):
            self.control = _parent_dataclass_from_child(GAMControl, self.control)

        self._train_index = X.index if isinstance(X, pd.DataFrame) else None
        nobs = X.shape[0]

        self._cox = isinstance(self.family, CoxFamily)
        if self._cox:
            family, response, offset, weight = self._cox_data(y, nobs)
            self._family = family
        else:
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

        self.terms_ = [deepcopy(term).setup(x) for term, x in zip(terms,
                                                                 self._columns(X, terms))]
        if linear_term is not None:
            linear_term = np.asarray(linear_term, float).reshape(-1)
            if linear_term.shape != (nobs,):
                raise ValueError(f'linear_term should have shape {(nobs,)}')

        regularizer = self.regularizer_ = GAMRegularizer(terms=self.terms_,
                                                         linear_term=linear_term,
                                                         bf_maxit=self.control.bf_maxit,
                                                         bf_tol=self.control.bf_tol,
                                                         modified=self.control.bf_modified,
                                                         fit_intercept=not self._cox)

        null_state, null_deviance = family.get_null_deviance(response=response,
                                                             sample_weight=normed_weight,
                                                             offset=offset,
                                                             fit_intercept=not self._cox)
        if self._cox:
            # no intercept: the null fit is the offset (as R's coxph)
            null_deviance = family.deviance(response,
                                            np.zeros(nobs) if offset is None else offset,
                                            normed_weight)
        self.null_deviance_ = weight_total * null_deviance
        null_intercept = 0. if self._cox else float(null_state.coef[0])

        def obj_function(state):
            return (family.deviance(response, state.mu, normed_weight) / 2 +
                    regularizer.objective(state))

        state = GAMState(np.zeros((nobs, len(self.terms_))), null_intercept)
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
        edf = []
        for j, term in enumerate(self.terms_):
            term.set_weights(w)
            edf.append(term.edf())
            term.finalize(state.fits[:, j], z - state.intercept - (total - state.fits[:, j]),
                          coefs[j])

        names = [term.name for term in self.terms_]
        self.state_ = state
        self.intercept_ = state.intercept
        self.fits_ = self._on_train_index_frame(state.fits, names)
        self.edf_ = pd.Series(edf, index=names, name='edf')

        self.deviance_ = weight_total * family.deviance(response, state.mu, normed_weight)
        # the intercept counts as one degree of freedom (none for Cox)
        self.df_resid_ = float(np.sum(weight > 0) - (not self._cox) - self.edf_.sum())

        self._estimated_dispersion = dispersion is None and not self._fixed_dispersion()
        if self._estimated_dispersion:
            dispersion = (np.sum(self._residuals('pearson')**2) / self.df_resid_
                          if self.df_resid_ > 0 else np.nan)
        elif dispersion is None:
            dispersion = 1.
        self.dispersion_ = dispersion

        # as R's gam (and glm): the dispersion, when free, counts as a
        # parameter, and the log-likelihood is at R's estimate of it
        k = (not self._cox) + self.edf_.sum() + (not self._fixed_dispersion())
        n = np.sum(weight > 0)
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
        The fitted function of a term: `term` is its index or its name (e.g.
        `'s(age, df=5)'`), and `x` its values, or a DataFrame of the data
        (with the term's columns).
        """
        if not isinstance(term, (int, np.integer)):
            term = [t.name for t in self.terms_].index(term)
        term = self.terms_[term]
        if isinstance(x, pd.DataFrame):
            x = self._columns(self._design(x), [term])[0]
        return term.predict(x)

    def _linear_predictor(self, X, offset=None):
        X = self._design(X)
        eta = self.intercept_ + sum(term.predict(x) for term, x in zip(self.terms_,
                                                                       self._columns(X, self.terms_)))
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
            if self._cox:
                return np.exp(eta)
            return self._family.base.link.inverse(eta)
        raise ValueError("prediction_type should be one of 'response' or 'link'")

    def _cox_data(self, y, nobs):
        """
        The Cox family of the event data in `y` (a DataFrame with the
        columns of `family`: event times, status, and optionally start times
        and strata), and the offset and weights (`offset_id`, `weight_id`).
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError('for the Cox family, y should be a DataFrame with the columns '
                             'of the event times and status (and the start times, strata, '
                             'offset and weights, if any)')
        if y.shape[0] != nobs:
            raise ValueError('X and y should have the same number of rows')
        fam = self.family
        family = CoxFamilySpec(event_data=y,
                               tie_breaking=fam.tie_breaking,
                               event_id=fam.event_id,
                               status_id=fam.status_id,
                               start_id=fam.start_id,
                               strata_id=fam.strata_id)
        offset = (None if self.offset_id is None
                  else np.asarray(y[self.offset_id], float).reshape(-1))
        weight = (np.ones(nobs) if self.weight_id is None
                  else np.asarray(y[self.weight_id], float).reshape(-1))
        return family, None, offset, weight

    def summary(self):
        """The terms, their degrees of freedom and smoothing parameters."""
        lam = [getattr(term, 'lam_', np.nan) for term in self.terms_]
        return pd.DataFrame({'edf': self.edf_.values, 'lam': lam},
                            index=self.edf_.index)

    # as for GLM: they use only the response, weights, family and state_
    _on_train_index = GLM._on_train_index
    _fixed_dispersion = GLM._fixed_dispersion

    def _residuals(self, kind='response', state=None):
        if getattr(self, '_cox', False):
            raise NotImplementedError('residuals of a Cox GAM are not available')
        return GLM._residuals(self, kind=kind, state=state)

    def _loglike(self):
        if self._cox:
            # the (log) partial likelihood, as R's coxph: the saturated one
            # less half the deviance
            result = self._family._coxdev_at(self._weight)(self.state_.link_parameter,
                                                                  self._weight)
            return float(result.loglik_sat - result.deviance / 2)
        base = self._family.base
        if isinstance(base, (sm_family.Gamma, sm_family.InverseGaussian)):
            # R's Gamma()$aic and inverse.gaussian()$aic: at the dispersion
            # deviance / sum(weights), not Pearson's
            y = np.asarray(self._response, float).reshape(-1)
            return base.loglike(y, self.state_.mu, var_weights=self._weight,
                                scale=self.deviance_ / self._weight.sum())
        return GLM._loglike(self)
