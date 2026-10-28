import warnings
from dataclasses import dataclass, field, InitVar
from typing import Optional, Literal
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt

from scipy.stats import norm as normal_dbn
from scipy.sparse.linalg import LinearOperator

from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

from coxdev import CoxDeviance, SurvivalCurves

    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)
from .scoring import Scorer
from .regularized_glm import RegGLM
from .glmnet import GLMNet
from ._utils import _get_data, _sub_ids, _check_offset, _check_predict_X
from ._roles import _role_y, cox_ids, with_sample_weight
from ._frames import as_pandas

_C_INDEX_DEPRECATED = ('c_index is deprecated, and will be removed: use coxdev.CoxDeviance('
                       'event, status, start=..., strata=..., sample_weight=...).c_index(pred)')
_COX_SURVFIT_DEPRECATED = ('cox_survfit is deprecated, and will be removed: use coxdev.CoxDeviance('
                           'event, status, start=..., strata=..., sample_weight=..., '
                           'tie_breaking=...).survfit(linear_predictor, ...)')


def c_index(pred,
            time,
            status,
            start=None,
            strata=None,
            sample_weight=None):
    """
    Deprecated: ``coxdev.CoxDeviance(...).c_index(pred)``.

    Harrell's concordance index for Cox model predictions, R's
    ``glmnet::Cindex`` (``survival::concordance(y ~ -pred, weights=w)``).
    R's ``Cindex`` ignores strata, so leave `strata` None to match it.

    Returns
    -------
    float or np.ndarray
        C index, one per column of `pred`; NaN if no pairs are comparable.
    """
    warnings.warn(_C_INDEX_DEPRECATED, FutureWarning, stacklevel=2)
    return CoxDeviance(event=time,
                       status=status,
                       strata=strata,
                       start=start,
                       sample_weight=sample_weight).c_index(pred, sample_weight=sample_weight)


@dataclass
class CoxSurvivalCurves(SurvivalCurves):
    """
    Survival curves from a Cox model, laid out as in R's ``survfit.coxph``:
    coxdev's ``SurvivalCurves`` (``CoxDeviance.survfit``), with `plot`.

    Rows are the distinct (stop) times within each stratum, stratum by
    stratum; columns are curves. A curve for a subject in one stratum is NaN
    on the rows of the other strata.

    Attributes
    ----------
    time : np.ndarray
        Times, shape `(ntimes,)`.
    stratum : np.ndarray
        Stratum label of each row (0 if the model is not stratified).
    n_risk, n_event, n_censor : np.ndarray
        Weighted number at risk, with an event, and censored at each time.
    cumulative_hazard : np.ndarray
        Cumulative hazard, shape `(ntimes, ncurves)`.
    stratified : bool
        Whether the model has strata.
    """
    stratified: bool = False

    @classmethod
    def from_curves(cls, curves, stratified):
        """The curves of ``CoxDeviance.survfit``, with `plot`."""
        return cls(time=curves.time,
                   stratum=curves.stratum,
                   n_risk=curves.n_risk,
                   n_event=curves.n_event,
                   n_censor=curves.n_censor,
                   cumulative_hazard=curves.cumulative_hazard,
                   stratified=stratified)

    @property
    def cumhaz(self):
        """Deprecated: `cumulative_hazard`."""
        warnings.warn('CoxSurvivalCurves.cumhaz is deprecated: use cumulative_hazard',
                      FutureWarning, stacklevel=2)
        return self.cumulative_hazard

    @property
    def surv(self):
        """Deprecated: `survival`."""
        warnings.warn('CoxSurvivalCurves.surv is deprecated: use survival',
                      FutureWarning, stacklevel=2)
        return self.survival

    @property
    def strata(self):
        """Deprecated: `stratum` (None if the model is not stratified)."""
        warnings.warn('CoxSurvivalCurves.strata is deprecated: use stratum (and stratified)',
                      FutureWarning, stacklevel=2)
        return self.stratum if self.stratified else None

    def plot(self, ax=None, **plot_args):
        """Plot each curve as a step function; returns the axes."""
        if ax is None:
            ax = plt.gca()
        H = self.cumulative_hazard
        for j in range(H.shape[1]):
            if self.stratified:
                groups = [self.stratum == s for s in pd.unique(self.stratum)]
            else:
                groups = [np.ones(self.time.shape[0], bool)]
            for rows in groups:
                rows = rows & ~np.isnan(H[:, j])
                if rows.any():
                    ax.step(np.r_[0, self.time[rows]], np.r_[1, self.survival[rows, j]],
                            where='post', **plot_args)
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival')
        return ax


def cox_survfit(linear_predictor,
                stop,
                status,
                start=None,
                strata=None,
                sample_weight=None,
                new_linear_predictor=None,
                new_strata=None,
                tie_breaking='efron',
                center=None):
    """
    Deprecated: ``coxdev.CoxDeviance(...).survfit(linear_predictor, ...)``.

    Survival curves for a Cox model with a fixed linear predictor: R's
    ``survfit.coxph`` (with ``se.fit=FALSE``) applied to ``coxph(y ~
    offset(linear_predictor), ties=tie_breaking)``.

    Returns
    -------
    CoxSurvivalCurves
    """
    warnings.warn(_COX_SURVFIT_DEPRECATED, FutureWarning, stacklevel=2)
    if tie_breaking not in ['efron', 'breslow']:
        raise ValueError("tie_breaking must be one of 'efron' or 'breslow'")
    cox = CoxDeviance(event=stop, status=status, start=start, strata=strata,
                      sample_weight=sample_weight, tie_breaking=tie_breaking)
    curves = cox.survfit(linear_predictor,
                         new_linear_predictor=new_linear_predictor,
                         new_strata=new_strata,
                         center=center)
    return CoxSurvivalCurves.from_curves(curves, stratified=strata is not None)

@dataclass
class CoxState(GLMState):
    """
    State for Cox regression models.
    
    Parameters
    ----------
    coef : np.ndarray
        Coefficient vector.
    obj_val : float, default=np.inf
        Objective function value.
    intercept : float, default=0
        Intercept term (always 0 for Cox models).
    """
    coef: np.ndarray
    obj_val: float = np.inf
    intercept: float = 0

    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset,
               objective=None):
        """
        Update the state with new design matrix and family.
        
        Parameters
        ----------
        design : np.ndarray
            Design matrix.
        family : CoxFamilySpec
            Cox family specification.
        offset : np.ndarray, optional
            Offset values.
        objective : callable, optional
            Objective function to evaluate.
        """
        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = self.link_parameter
        
        # shorthand
        self.mu = self.link_parameter
        self.eta = self.linear_predictor
        
        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):
        """
        Compute the log-likelihood score.
        
        Parameters
        ----------
        family : CoxFamilySpec
            Cox family specification.
        y : np.ndarray
            Response variable.
        sample_weight : np.ndarray
            Sample weights.
            
        Returns
        -------
        np.ndarray
            Log-likelihood score.
        """
        link_parameter = self.link_parameter
        family._result = family._coxdev_at(sample_weight)(link_parameter,
                                                          sample_weight)
        # the gradient is the gradient of the deviance
        # we want gradient of the log-likelihood
        return - family._result.gradient / 2

@dataclass
class CoxFamily(object):
    """
    Cox family specification for basic configuration.
    
    Parameters
    ----------
    tie_breaking : {'breslow', 'efron'}, default='breslow'
        Method for handling ties in survival times. The default is
        'breslow', as R's glmnet (``cox.ties="breslow"``).
    event_id : str, optional, default='event'
        Column name for event times.
    status_id : str, optional, default='status'
        Column name for event status (0=censored, 1=event).
    start_id : str, optional, default=None
        Column name for start times (for start-stop data).
    strata_id : str, optional, default=None
        Column name for strata (for stratified Cox models).

    The ids are deprecated: name the columns of `y` by their role, `stop`,
    `status`, `start` and `strata`, e.g. `Y = Outcome(Surv('time',
    'status'), strata='inst').fit_transform(D)` (modelmatrix).
    """
    tie_breaking: Literal['breslow', 'efron'] = 'breslow'
    event_id: Optional[str] = 'event'
    status_id: Optional[str] = 'status'
    start_id: Optional[str] = None
    strata_id: Optional[str] = None

@dataclass
class CoxFamilySpec(object):
    """
    Cox family specification for survival analysis.
    
    Parameters
    ----------
    event_data : InitVar[np.ndarray]
        Survival data containing event times, status, and optionally start times.
    tie_breaking : {'breslow', 'efron'}, default='breslow'
        Method for handling ties in survival times. The default is
        'breslow', as R's glmnet (``cox.ties="breslow"``).
    event_id : str, optional, default='event'
        Column name for event times.
    status_id : str, optional, default='status'
        Column name for event status (0=censored, 1=event).
    start_id : str, optional, default=None
        Column name for start times (for start-stop data).
    strata_id : str, optional, default=None
        Column name for strata (for stratified Cox models).
    name : str, default='Cox'
        Family name.
    """
    event_data: InitVar[np.ndarray]
    tie_breaking: Literal['breslow', 'efron'] = 'breslow'
    event_id: Optional[str] = 'event'
    status_id: Optional[str] = 'status'
    start_id: Optional[str] = None
    strata_id: Optional[str] = None
    name: str = 'Cox'
    
    def __hash__(self):
        return (self.tie_breaking,
                self.event_id,
                self.status_id,
                self.start_id,
                self.strata_id,
                self.name).__hash__()

    def __post_init__(self, event_data):
        self.is_gaussian = False
        self.is_binomial = False

        if (self.event_id not in event_data.columns or
            self.status_id not in event_data.columns):
            raise ValueError(f'expecting f{self.event_id} and f{self.status_id} columns')
        
        event = event_data[self.event_id]
        status = event_data[self.status_id]
        n = len(event)

        if self.strata_id is not None and self.strata_id in event_data.columns:
            # coxdev requires integer strata labels
            strata = pd.factorize(event_data[self.strata_id], sort=True)[0]
        else:
            strata = np.zeros(n, dtype=int)
        self.strata = strata

        start = event_data[self.start_id] if self.start_id is not None else None
        self._coxdev_args = dict(event=np.asarray(event, float),
                                 status=status,
                                 start=None if start is None else np.asarray(start, float),
                                 strata=strata,
                                 tie_breaking=self.tie_breaking)
        self._coxdev_ = None

    def _coxdev_at(self, sample_weight):
        """
        The CoxDeviance of the data, built on first use with `sample_weight`:
        it sorts the data again whenever it is called with other weights, so
        building it with the weights of its first call saves one sort.
        """
        if self._coxdev_ is None:
            self._coxdev_ = CoxDeviance(**self._coxdev_args, sample_weight=sample_weight)
        return self._coxdev_

    @property
    def _coxdev(self):
        return self._coxdev_at(None)

    # GLMFamilySpec API
    def link(self, mu):
        return mu

    def predict(self, linpred, prediction_type='response'):
        # Cox predictions are on the linear predictor scale, as in CoxNet.predict
        return linpred

    def deviance(self, 
                 y,
                 mu,
                 sample_weight=None):

        link_parameter = mu
        self._result = self._coxdev_at(sample_weight)(link_parameter,
                                                      sample_weight)
        if np.isnan(self._result.deviance):
            raise ValueError
        return self._result.deviance
    
    def null_fit(self,
                 y,
                 sample_weight,
                 fit_intercept):
        sample_weight = np.asarray(sample_weight)
        return np.zeros_like(sample_weight)

    def get_null_deviance(self,
                          response,
                          sample_weight,
                          offset, # ignored for Cox
                          fit_intercept):
        mu0 = self.null_fit(response, sample_weight, fit_intercept)
        return mu0, self.deviance(response, mu0, sample_weight)

    def _get_null_state(self,
                        null_fit,
                        nvars):
        coefold = np.zeros(nvars)   # initial coefs = 0
        return CoxState(coef=coefold,
                        intercept=0)

    def get_response_and_weights(self,
                                 state,
                                 y,
                                 offset,
                                 sample_weight=None):

        link_parameter = state.link_parameter
        linear_predictor = state.linear_predictor
        self._result = self._coxdev_at(sample_weight)(link_parameter,
                                                      sample_weight)
        # self._coxdev computes value, gradient and hessian of deviance
        # we want the gradient, hessian of deviance / 2
        gradient = self._result.gradient / 2
        diag_hessian = self._result.diag_hessian / 2
        test = diag_hessian != 0
        newton_weights = diag_hessian
        inv_weights = np.where(test, 1 / (diag_hessian + (1 - test)), 0)
        pseudo_response = linear_predictor - gradient * inv_weights

        return pseudo_response, newton_weights
    
    def information(self,
                    state,
                    sample_weight):
        info = self._coxdev_at(sample_weight).information(state.link_parameter,
                                                          sample_weight)
        if not hasattr(info, '_xp'):
            # coxdev <= 0.1.6 does not call LinearOperator.__init__, which
            # SciPy >= 1.18 needs (it sets the array namespace used by @)
            LinearOperator.__init__(info, dtype=info.dtype, shape=info.shape)
        return info

    sample_score_methods = ('multinomial',
                            'bernoulli',
                            'fisher',
                            'wild')

    def sample_score(self,
                     linear_predictor,
                     method='multinomial',
                     n_draws=None,
                     sample_weight=None,
                     rng=None):
        """
        Draws of the score at a linear predictor, computed by `coxdev`.

        Each draw ``r`` has one entry per observation, on the scale of the
        score (minus half the gradient of the deviance), so that ``X.T @ r``
        is a draw of the score of the coefficients of a design ``X``. With
        `sample_weight` None (all one), that is the total score.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor (with any offset), e.g. of a pilot fit.
        method : {'multinomial', 'bernoulli', 'fisher', 'wild'}
            Passed to `coxdev`, whose samplers make the draws (see its
            documentation for details and references). The model-based
            methods draw the failures at each event time from the model and
            return their score, centered at its expectation:

            - ``'multinomial'``: with replacement from the risk set, with
              probabilities proportional to ``sample_weight *
              exp(linear_predictor)``. Its covariance is the Breslow
              information.
            - ``'bernoulli'``: each subject at risk fails independently with
              probability ``min(1, d p)``.
            - ``'fisher'``: the set of failures from Fisher's noncentral
              hypergeometric distribution. Its covariance is the information
              of the exact discrete partial likelihood.
            - ``'wild'``: the score residuals (with Efron's weights if
              ``tie_breaking='efron'``) times independent standard normals.
              Its covariance is the sandwich, which does not need the model
              to be right.
        n_draws : int, optional
            The number of draws. If None, one draw, as a vector.
        sample_weight : np.ndarray, optional
            Observation weights, default all one. All methods but ``'wild'``
            need equal positive weights (zero weights drop out).
        rng : numpy Generator (or seed), optional
            Source of the seed passed to `coxdev`, whose own generator makes
            the draws.

        Returns
        -------
        np.ndarray
            ``(n,)`` if `n_draws` is None, else ``(n, n_draws)``.
        """
        if method not in self.sample_score_methods:
            raise ValueError(f'method must be one of {self.sample_score_methods}, got {method!r}')
        coxdev = self._coxdev_at(sample_weight)
        if not hasattr(coxdev, 'sample_score'):
            raise ImportError('sample_score needs a version of coxdev with CoxDeviance.sample_score')
        rng = np.random.default_rng(rng)
        return coxdev.sample_score(np.asarray(linear_predictor, float),
                                   sample_weight=sample_weight,
                                   method=method,
                                   n_draws=n_draws,
                                   seed=int(rng.integers(2**63)))

    def _default_scorers(self):

        return [CoxDiffScorer(coxfam=self), CoxScorer(coxfam=self)]

def _cox_score(estimator, X, y, sample_weight=None):
    """
    The partial log-likelihood (minus half the deviance) of a fitted Cox
    `estimator` on `(X, y)`: its family holds the event data of the fit, so
    the family of `y` is built for scoring new data (as `GridSearchCV` does).
    """
    X, y = as_pandas(X), as_pandas(y)
    y = with_sample_weight(_role_y(estimator, y), sample_weight)
    family = estimator._finalize_family(response=y)
    _, _, _, offset, weight = estimator.get_data_arrays(X, y, check=False)
    eta = estimator._linear_predictor(X, offset)
    return -family.deviance(None, eta, np.asarray(weight, float)) / 2


@dataclass
class CoxLM(GLM):
    """
    Cox Linear Model for survival analysis.
    
    Fits a Cox proportional hazards model without regularization.
    
    Parameters
    ----------
    fit_intercept : Literal[False], default=False
        Whether to fit an intercept. For Cox models, this is always False
        as the intercept is absorbed into the baseline hazard.
    family : CoxFamily
        The tie breaking and the columns of the survival data in `y`.
    """
    fit_intercept: Literal[False] = False
    family: CoxFamily = field(default_factory=CoxFamily)

    _shift_invariant = True

    def _finalize_family(self,
                         response):
        response = _role_y(self, response)
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             **cox_ids(response))

    def _get_regularizer(self,
                         nvars=None):
        # the unpenalized GLM's regularizer, taking Newton steps with the
        # full information (see _newton_step)
        regularizer = super()._get_regularizer(nvars=nvars)
        regularizer.newton_step = self._newton_step
        return regularizer

    def _newton_step(self,
                     design,
                     pseudo_response,              # ignored
                     sample_weight,                # ignored: the diagonal of the Hessian
                     cur_state):
        """
        A Newton step from `cur_state` with the full information.

        The IRLS step of `GLMRegularizer` weights the observations by the
        diagonal of the Hessian of the partial likelihood in the linear
        predictor. For a GLM that is the Hessian, but for Cox the Hessian is
        not diagonal, and IRLS converges only linearly (slowly: it may stop at
        `mxitnr` far from the maximum). This step uses the full information
        `X^T H X` (from `coxdev`), as R's `coxph`; `quasi_newton_step` halves
        it if the objective does not decrease. `GLM.fit` has set the family
        and the weights by the time it is called.

        Parameters
        ----------
        design: Design
            Design matrix.
        pseudo_response: np.ndarray
            Ignored.
        sample_weight: np.ndarray
            Ignored (IRLS's working weights).
        cur_state: CoxState
            Current state.

        Returns
        -------
        CoxState
            Updated state.
        """
        w = self.sample_weight_
        # the score and information in the linear predictor, at the weights of the fit
        score = cur_state.logl_score(self._family, None, w)
        info = self._family.information(cur_state, w)

        # in the coordinates of the state (the design's), without the
        # intercept (Cox has none) and the excluded variables
        p = design.X.shape[1]
        keep = np.ones(p, bool)
        keep[np.asarray(self._excluded(), int)] = False
        kept = np.nonzero(keep)[0]

        coefnew = np.zeros(p)
        if kept.shape[0] > 0:
            coef = np.asarray(cur_state.coef)[kept]
            H = design.quadratic_form(info, transformed=True)[1:, 1:][kept][:, kept]
            g = (design.T @ score)[1:][kept]
            if self.ridge_coef != 0:
                H = H + self.ridge_coef * np.identity(H.shape[0])
                g = g - self.ridge_coef * coef
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ g
            coefnew[kept] = coef + delta

        # as GLMRegularizer.newton_step, the step is the regularizer's warm state
        self.regularizer_.warm_state = cur_state.__class__(coefnew, 0)
        return self.regularizer_.warm_state

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        return _get_data(self,
                         X,
                         y,
                         check=check,
                         multi_output=True)

    def _loglike(self):
        # the partial log-likelihood
        result = self._family._coxdev_at(self._weight)(self.state_.mu, self._weight)
        return result.loglik_sat - result.deviance / 2

    def _score_meat(self):
        S = self._score_contributions()
        return S.T @ S

    def _score_contributions(self):
        # the score residuals (Lin and Wei, 1989), as R's coxph(robust=TRUE)
        design = self.design_
        X = design.X[:, np.nonzero(self._kept)[0]]
        coxdev = self._family._coxdev_at(self.sample_weight_)
        U = coxdev.score_residuals(self.state_.link_parameter, X,
                                   sample_weight=self.sample_weight_)
        U = U.reshape((U.shape[0], -1))
        # a dense design is stored standardized; the score residuals are
        # centered in each risk set, so only its scaling matters
        if not scipy.sparse.issparse(design.X):
            U = U * design.scaling_[self._kept][None, :]
        # they are linear in the weights, which are normalized
        return self.weight_total_ * U

    def score(self, X, y, sample_weight=None):
        """
        Partial log-likelihood (minus half the deviance) of the fit on `(X, y)`,
        with the weights of `y` (its column `weight`, or `sample_weight` in its place).
        """
        return _cox_score(self, X, y, sample_weight)

    def _residuals(self, kind='martingale', state=None):
        """The residuals of `glmnet.glm.residuals`: martingale or deviance, at `state_`."""
        w = np.asarray(self._weight, float)
        result = self._family._coxdev_at(w)(self.state_.mu, w)
        # the deviance's gradient in the linear predictor is -2 w M, with M
        # the martingale residuals
        M = np.where(w > 0, -result.gradient / 2 / np.where(w > 0, w, 1), 0)
        if kind == 'martingale':
            value = M
        elif kind == 'deviance':
            status = np.asarray(self._family._coxdev_args['status'], float)
            with np.errstate(divide='ignore', invalid='ignore'):
                d = -2 * (M + np.where(status > 0, status * np.log(status - M), 0))
            value = np.sign(M) * np.sqrt(np.maximum(d, 0))
        else:
            raise ValueError("Cox residuals are 'martingale' or 'deviance'")
        return self._on_train_index(value, kind)


@dataclass
class RegCoxLM(RegGLM):
    """
    Regularized Cox Linear Model for survival analysis.
    
    Fits a Cox proportional hazards model with regularization (lasso, ridge, or elastic net).
    
    Parameters
    ----------
    fit_intercept : Literal[False], default=False
        Whether to fit an intercept. For Cox models, this is always False
        as the intercept is absorbed into the baseline hazard.
    family : CoxFamily
        The tie breaking and the columns of the survival data in `y`.
    """
    fit_intercept: Literal[False] = False
    family: CoxFamily = field(default_factory=CoxFamily)

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        return _get_data(self,
                         X,
                         y,
                         check=check,
                         multi_output=True)

    def _finalize_family(self,
                         response):
        response = _role_y(self, response)
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             **cox_ids(response))

    def score(self, X, y, sample_weight=None):
        """
        Partial log-likelihood (minus half the deviance) of the fit on `(X, y)`,
        with the weights of `y` (its column `weight`, or `sample_weight` in its place).
        """
        return _cox_score(self, X, y, sample_weight)

@dataclass
class CoxNetIRLS(GLMNet):
    """
    CoxNetIRLS: Cox Proportional Hazards Model with Elastic Net regularization.
    
    Fits a Cox proportional hazards model with regularization along a path of lambda values.
    Supports both right-censored and start-stop survival data with Breslow or Efron tie-breaking.
    
    The path is computed by IRLS in Python around the generic `GLMNet` solver.
    `glmnet.CoxNet` (`glmnet.paths.CoxNet`) fits the same model with the C++
    Cox path used by R's glmnet.

    Parameters
    ----------
    family : CoxFamily, default=CoxFamily()
        Column names for the survival data and tie-breaking method.
    fit_intercept : Literal[False], default=False
        Whether to fit an intercept. For Cox models, this is always False
        as the intercept is absorbed into the baseline hazard.
    regularized_estimator : BaseEstimator, default=RegCoxLM
        The regularized estimator class to use for fitting.
    """
    family: CoxFamily = field(default_factory=CoxFamily)
    fit_intercept: Literal[False] = False
    regularized_estimator: BaseEstimator = RegCoxLM
    
    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        return _get_data(self,
                         X,
                         y,
                         check=check,
                         multi_output=True)

    def _finalize_family(self,
                         response):
        response = _role_y(self, response)
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             **cox_ids(response))
    
    def get_LM(self):
        return CoxLM(family=self.family,
                     **_sub_ids(self))

    def _get_initial_state(self,
                           X,
                           y,
                           exclude):

        n, p = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor_ == 0
        keep[exclude] = 0

        coef_ = np.zeros(p)
        intercept_ = 0

        if keep.sum() > 0:
            X_keep = X[:,keep]

            coxlm = self.get_LM()
            coxlm.fit(X_keep, y)
            coef_[keep] = coxlm.coef_

        return CoxState(coef_, intercept_), keep.astype(float)

    def predict(self,
                X,
                prediction_type='link',
                interpolation_grid=None,
                offset=None,
                gamma=1.):
        """
        Predict using the fitted CoxNetIRLS model.

        Parameters
        ----------
        X : Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized. If it is not a sparse matrix, a copy is made and
            standardized.
        prediction_type : {'link', 'response'}, default='link'
            As R's `predict.coxnet`: 'link' gives the linear predictor (risk
            score), 'response' the relative risk `exp(link)`.
        interpolation_grid : np.ndarray, optional
            Grid of lambda values for interpolation. If provided, coefficients are 
            interpolated to these values before prediction.
        offset : np.ndarray, optional
            Offset for the rows of `X`, of shape `(n_samples,)`, added to the
            linear predictor (R's `newoffset`). If omitted, no offset is used.
        gamma : float, optional
            Blend of the lasso (1, the default) and relaxed (0) fits, as R's
            `predict(..., gamma=)`; requires `relax=True` unless 1.

        Returns
        -------
        np.ndarray
            Predictions for each lambda value. Shape is (n_samples, n_lambdas)
            where n_lambdas is the number of lambda values in the fitted path
            or the length of interpolation_grid if provided.
        """
        X = as_pandas(X)
        if prediction_type not in ['link', 'response']:
            raise ValueError("prediction_type should be one of 'link' or 'response' for Cox models")
        X = _check_predict_X(X)

        if interpolation_grid is not None:
            grid_ = np.asarray(interpolation_grid)
            coefs_, intercepts_ = self.interpolate_coefs(grid_, gamma=gamma)
        else:
            grid_ = None
            coefs_, intercepts_ = self._blended_coefs(gamma)

        intercepts_ = np.atleast_1d(intercepts_)
        coefs_ = np.atleast_2d(coefs_)
        linear_pred_ = coefs_ @ X.T + intercepts_[:, None]
        linear_pred_ = linear_pred_.T
        if offset is not None:
            linear_pred_ = linear_pred_ + _check_offset(offset, X.shape[0])[:, None]

        # make return based on original
        # promised number of lambdas
        # pad with last value
        if grid_ is not None:
            if grid_.shape:
                nlambda = coefs_.shape[0]
                squeeze = False
            else:
                nlambda = 1
                squeeze = True
        else:
            if self.lambda_values is not None:
                nlambda = self.lambda_values.shape[0]
            else:
                nlambda = self.nlambda
            squeeze = False

        if prediction_type == 'response':
            linear_pred_ = np.exp(linear_pred_)

        value = np.zeros((linear_pred_.shape[0], nlambda), float) * np.nan
        value[:,:linear_pred_.shape[1]] = linear_pred_
        value[:,linear_pred_.shape[1]:] = linear_pred_[:,-1][:,None]
        
        if squeeze:
            value = np.squeeze(value)
        return value

    
@dataclass(frozen=True)
class CoxScorer(Scorer):

    coxfam: CoxFamilySpec=None
    use_full_data: bool=True
    maximize: bool=False

    name: str = 'Cox Deviance'

    def score_fn(self,
                 split,
                 event_data,
                 predictions,
                 sample_weight):
        
        coxfam = self.coxfam
        sample_weight = np.asarray(sample_weight)
        status = np.asarray(event_data[coxfam.status_id])

        cls = self.coxfam.__class__

        predictions = np.asarray(predictions)
        event_split = event_data.iloc[split]
        fam_split = cls(tie_breaking=coxfam.tie_breaking,
                        event_id=coxfam.event_id,
                        status_id=coxfam.status_id,
                        start_id=coxfam.start_id,
                        strata_id=coxfam.strata_id,
                        event_data=event_split)

        split_w = sample_weight[split]
        dev_split = fam_split._coxdev_at(split_w)(predictions[split], split_w).deviance
        w_sum = sample_weight[split].sum()

        return dev_split / w_sum, w_sum

@dataclass(frozen=True)
class CoxDiffScorer(CoxScorer):

    name: str = 'Cox Deviance (Difference)'
    def score_fn(self,
                 split,
                 event_data,
                 predictions,
                 sample_weight):
        
        coxfam = self.coxfam
        sample_weight = np.asarray(sample_weight)
        status = np.asarray(event_data[coxfam.status_id])
        cls = self.coxfam.__class__
        predictions = np.asarray(predictions)

        fam_full = cls(tie_breaking=coxfam.tie_breaking,
                       event_id=coxfam.event_id,
                       status_id=coxfam.status_id,
                       start_id=coxfam.start_id,
                       strata_id=coxfam.strata_id,
                       event_data=event_data)
        dev_full = fam_full._coxdev_at(sample_weight)(predictions, sample_weight).deviance

        # now compute deviance on complement
        
        split_c = np.ones_like(predictions, bool)
        split_c[split] = 0

        event_c = event_data.iloc[split_c] # XXX presumes dataframe, could be ndarray
        fam_split_c = cls(tie_breaking=coxfam.tie_breaking,
                          event_id=coxfam.event_id,
                          status_id=coxfam.status_id,
                          start_id=coxfam.start_id,
                          strata_id=coxfam.strata_id,
                          event_data=event_c)
        split_c_w = sample_weight[split_c]
        dev_c = fam_split_c._coxdev_at(split_c_w)(predictions[split_c], split_c_w).deviance

        w_sum = sample_weight.sum() - split_c_w.sum()
        return (dev_full - dev_c) / w_sum, w_sum


    




  
@dataclass(frozen=True)
class CoxCIndexScorer(Scorer):
    """
    Harrell's C index (R's ``type.measure="C"``), computed per fold and
    averaged with fold weights equal to the summed sample weights, as in R's
    ``cv.coxnet``. Strata are ignored unless ``stratified=True``, matching R.

    The columns are those of `y` named by their role, `stop`, `status`, and
    `start` and `strata` if `y` has them; ``CoxCIndexScorer.from_family(family)``
    takes the column names of a `CoxFamily` or `CoxFamilySpec` instead.
    """

    name: str = 'C-index'
    maximize: bool = True
    use_full_data: bool = True
    event_id: str = 'stop'
    status_id: str = 'status'
    start_id: Optional[str] = 'start'      # (if the data have it)
    strata_id: Optional[str] = 'strata'    # (if the data have it)
    stratified: bool = False

    @staticmethod
    def from_family(family, **kwargs):
        return CoxCIndexScorer(event_id=family.event_id,
                               status_id=family.status_id,
                               start_id=family.start_id,
                               strata_id=family.strata_id,
                               **kwargs)

    def score_fn(self,
                 split,
                 event_data,
                 predictions,
                 sample_weight):

        event_split = event_data.iloc[split]
        start = strata = None
        if self.start_id is not None and self.start_id in event_split.columns:
            start = event_split[self.start_id]
        if (self.stratified and self.strata_id is not None
            and self.strata_id in event_split.columns):
            strata = event_split[self.strata_id]
        split_w = np.asarray(sample_weight)[split]

        cox = CoxDeviance(event=event_split[self.event_id],
                          status=event_split[self.status_id],
                          start=start,
                          strata=strata,
                          sample_weight=split_w)
        value = cox.c_index(np.asarray(predictions)[split], sample_weight=split_w)
        return value, split_w.sum()
