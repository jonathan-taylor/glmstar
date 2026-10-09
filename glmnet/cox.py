from dataclasses import dataclass, field, InitVar
from typing import Optional, Literal
from functools import partial

import numpy as np
import pandas as pd

from scipy.stats import norm as normal_dbn
from scipy.sparse.linalg import LinearOperator

from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

from coxdev import CoxDeviance

from ._concordance import concordance as _concordance
    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)
from .scoring import Scorer
from .regularized_glm import RegGLM
from .glmnet import GLMNet
from ._utils import _get_data, _check_offset

def c_index(pred,
            time,
            status,
            start=None,
            strata=None,
            sample_weight=None):
    """
    Harrell's concordance index for Cox model predictions.

    Matches R's ``glmnet::Cindex``, i.e. ``survival::concordance(y ~ -pred,
    weights=w)``: for each event time ``t`` an event ``i`` is compared to
    every ``j`` at risk at ``t`` (``start_j < t <= stop_j``), except other
    events tied at ``t``. A pair is concordant if ``pred_i > pred_j``, counts
    1/2 if the predictions are tied, and has weight ``w_i * w_j``.

    Computed in O(n log n) per column, as in the survival package's
    concordance vignette: subjects are processed in decreasing time order,
    with the risk set's weights kept in a binary indexed tree over the ranks
    of `pred`.

    Parameters
    ----------
    pred : np.ndarray
        Linear predictor (risk score), of shape `(n,)` or `(n, nlambda)`.
    time : np.ndarray
        Event (stop) times.
    status : np.ndarray
        Event indicator (1=event, 0=censored).
    start : np.ndarray, optional
        Start times for (start, stop] data.
    strata : np.ndarray, optional
        If given, only pairs within the same stratum are compared. R's
        ``Cindex`` ignores strata, so leave as None to match it.
    sample_weight : np.ndarray, optional
        Observation weights.

    Returns
    -------
    float or np.ndarray
        C index, one per column of `pred`; NaN if no pairs are comparable.
    """
    pred = np.asarray(pred, float)
    squeeze = pred.ndim == 1
    pred = pred.reshape((pred.shape[0], -1))
    time = np.asarray(time, float)
    status = np.asarray(status).astype(np.int32)
    n = time.shape[0]
    start = np.full(n, -np.inf) if start is None else np.asarray(start, float)
    if strata is None:
        strata = np.zeros(n, np.int32)
    else:
        strata = pd.factorize(np.asarray(strata))[0].astype(np.int32)
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)

    num, den = _concordance(np.asfortranarray(pred), time, status, start, strata, w)
    value = num / den if den > 0 else np.full(pred.shape[1], np.nan)
    return value[0] if squeeze else value

@dataclass
class CoxSurvivalCurves(object):
    """
    Survival curves from a Cox model, laid out as in R's ``survfit.coxph``.

    Rows are the distinct (stop) times within each stratum, stratum by
    stratum; columns are curves. A curve for a subject in one stratum is NaN
    on the rows of the other strata.

    Attributes
    ----------
    time : np.ndarray
        Times, shape `(ntimes,)`.
    strata : np.ndarray or None
        Stratum label of each row (None if the model is not stratified).
    n_risk, n_event, n_censor : np.ndarray
        Weighted number at risk, with an event, and censored at each time.
    cumhaz : np.ndarray
        Cumulative hazard, shape `(ntimes, ncurves)`.
    """
    time: np.ndarray
    strata: Optional[np.ndarray]
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censor: np.ndarray
    cumhaz: np.ndarray

    @property
    def surv(self):
        """Survival probabilities exp(-cumhaz), as R's default ``stype=2``."""
        return np.exp(-self.cumhaz)

    def predict(self, times):
        """
        Evaluate the curves at the given times.

        Parameters
        ----------
        times : np.ndarray
            Times at which to evaluate, shape `(m,)`.

        Returns
        -------
        np.ndarray
            Survival probabilities, shape `(m, ncurves)`. Each curve is a
            right-continuous step function, equal to 1 before its first time.
        """
        times = np.atleast_1d(np.asarray(times, float))
        value = np.empty((times.shape[0], self.cumhaz.shape[1]))
        for j in range(self.cumhaz.shape[1]):
            rows = np.nonzero(~np.isnan(self.cumhaz[:, j]))[0]
            pos = np.searchsorted(self.time[rows], times, side='right') - 1
            H = np.where(pos >= 0, self.cumhaz[rows[np.maximum(pos, 0)], j], 0)
            value[:, j] = np.exp(-H)
        return value

    def plot(self, ax=None, **plot_args):
        """Plot each curve as a step function; returns the axes."""
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        for j in range(self.cumhaz.shape[1]):
            if self.strata is None:
                groups = [np.ones(self.time.shape[0], bool)]
            else:
                groups = [self.strata == s for s in pd.unique(self.strata)]
            for rows in groups:
                rows = rows & ~np.isnan(self.cumhaz[:, j])
                if rows.any():
                    ax.step(np.r_[0, self.time[rows]], np.r_[1, self.surv[rows, j]],
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
    Survival curves for a Cox model with a fixed linear predictor.

    Matches R's ``survfit.coxph`` (with ``se.fit=FALSE``) applied to
    ``coxph(y ~ offset(linear_predictor), ties=tie_breaking)``, which is how
    R's ``survfit.coxnet`` computes curves. The baseline cumulative hazard is
    the Breslow estimator, or its Efron-corrected version for tied events.

    Parameters
    ----------
    linear_predictor : np.ndarray
        Linear predictor (including any offset) for the training data.
    stop, status : np.ndarray
        Event (stop) times and event indicator (1=event, 0=censored).
    start : np.ndarray, optional
        Start times for (start, stop] data.
    strata : np.ndarray, optional
        Stratum labels; each stratum has its own baseline hazard.
    sample_weight : np.ndarray, optional
        Observation weights.
    new_linear_predictor : np.ndarray, optional
        Linear predictors of the subjects to compute curves for. If None,
        a single curve is computed at the weighted mean linear predictor,
        as R does when ``newdata`` is missing.
    new_strata : np.ndarray, optional
        Stratum labels of the new subjects; required if `strata` is given
        along with `new_linear_predictor`.
    tie_breaking : {'efron', 'breslow'}
        Hazard estimate for tied event times (R's ``ctype`` 2 or 1).
    center : float, optional
        Linear predictor of the curve computed when `new_linear_predictor` is
        None. Defaults to the weighted mean of `linear_predictor`.

    Returns
    -------
    CoxSurvivalCurves
    """
    lp = np.asarray(linear_predictor, float)
    stop = np.asarray(stop, float)
    status = np.asarray(status).astype(bool)
    n = stop.shape[0]
    start = np.full(n, -np.inf) if start is None else np.asarray(start, float)
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    if tie_breaking not in ['efron', 'breslow']:
        raise ValueError("tie_breaking must be one of 'efron' or 'breslow'")

    # all hazards are computed for linear predictor `center` and then rescaled
    if center is None:
        center = np.sum(w * lp) / np.sum(w)
    if new_linear_predictor is None:
        new_lp = np.array([center])
    else:
        new_lp = np.atleast_1d(np.asarray(new_linear_predictor, float))

    if strata is None:
        labels = np.zeros(n, int)
        levels = np.array([0])
        new_labels = None
    else:
        labels, levels = pd.factorize(np.asarray(strata), sort=True)
        if new_linear_predictor is None:
            new_labels = None
        else:
            if new_strata is None:
                raise ValueError('new_strata is required for curves from a stratified model')
            new_labels = pd.Index(levels).get_indexer(np.atleast_1d(np.asarray(new_strata)))
            if np.any(new_labels < 0):
                raise ValueError('new_strata contains labels not seen in the training data')
            if new_labels.shape[0] != new_lp.shape[0]:
                raise ValueError('new_strata and new_linear_predictor have different lengths')

    risk = w * np.exp(lp - center)
    results = []
    for s in range(len(levels)):
        idx = labels == s
        stop_s, status_s, start_s = stop[idx], status[idx], start[idx]
        w_s, risk_s = w[idx], risk[idx]
        times = np.unique(stop_s)

        # sums over {j: start_j < t <= stop_j} = {stop_j >= t} - {start_j >= t}
        def at_risk(v):
            by_stop, by_start = np.argsort(stop_s), np.argsort(start_s)
            tail_stop = np.r_[np.cumsum(v[by_stop][::-1])[::-1], 0]
            tail_start = np.r_[np.cumsum(v[by_start][::-1])[::-1], 0]
            return (tail_stop[np.searchsorted(stop_s[by_stop], times, 'left')] -
                    tail_start[np.searchsorted(start_s[by_start], times, 'left')])

        n_risk = at_risk(w_s)
        risk_sum = at_risk(risk_s)

        pos = np.searchsorted(times, stop_s)
        n_event = np.bincount(pos, weights=w_s * status_s, minlength=len(times))
        n_censor = np.bincount(pos, weights=w_s * ~status_s, minlength=len(times))
        d = np.bincount(pos, weights=status_s.astype(float), minlength=len(times))
        event_risk = np.bincount(pos, weights=risk_s * status_s, minlength=len(times))

        with np.errstate(divide='ignore', invalid='ignore'):
            dH = np.where(n_event > 0, n_event / risk_sum, 0)
        if tie_breaking == 'efron':
            for k in np.nonzero(d > 1)[0]:
                frac = np.arange(d[k]) / d[k]
                dH[k] = np.sum(n_event[k] / d[k] / (risk_sum[k] - frac * event_risk[k]))
        H = np.cumsum(dH)

        if new_labels is None:
            cumhaz = H[:, None] * np.exp(new_lp - center)[None, :]
        else:
            cumhaz = np.full((len(times), new_lp.shape[0]), np.nan)
            in_s = new_labels == s
            cumhaz[:, in_s] = H[:, None] * np.exp(new_lp[in_s] - center)[None, :]
        results.append((times, np.full(len(times), levels[s]), n_risk, n_event, n_censor, cumhaz))

    time_, strata_, n_risk_, n_event_, n_censor_, cumhaz_ = [np.concatenate(v) for v in zip(*results)]
    return CoxSurvivalCurves(time=time_,
                             strata=None if strata is None else strata_,
                             n_risk=n_risk_,
                             n_event=n_event_,
                             n_censor=n_censor_,
                             cumhaz=cumhaz_)

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
        family._result = family._coxdev(link_parameter,
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
        
        if self.start_id is not None:
            start = event_data[self.start_id]
            self._coxdev = CoxDeviance(
                np.asarray(event, float),
                status,
                start=np.asarray(start, float),
                strata=strata,
                tie_breaking=self.tie_breaking
            )
        else:
            start = None
            self._coxdev = CoxDeviance(
                np.asarray(event, float),
                status,
                start=None,
                strata=strata,
                tie_breaking=self.tie_breaking
            )

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
        self._result = self._coxdev(link_parameter,
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
        self._result = self._coxdev(link_parameter,
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
        info = self._coxdev.information(state.link_parameter,
                                        sample_weight)
        if not hasattr(info, '_xp'):
            # coxdev <= 0.1.6 does not call LinearOperator.__init__, which
            # SciPy >= 1.18 needs (it sets the array namespace used by @)
            LinearOperator.__init__(info, dtype=info.dtype, shape=info.shape)
        return info

    def _default_scorers(self):

        return [CoxDiffScorer(coxfam=self), CoxScorer(coxfam=self)]

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
    """
    fit_intercept: Literal[False] = False

    def _finalize_family(self,
                         response):
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id,
                             strata_id=self.family.strata_id)

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   X_shape):

        # IRLS used normalized weights,
        # this unnormalizes them...

        unscaled_precision_ = self.design_.quadratic_form(self._information,
                                                          transformed=False)
        
        keep = np.ones(unscaled_precision_.shape[0]-1, bool)
        if exclude is not []:
            keep[exclude] = 0
        keep = np.hstack([self.fit_intercept, keep]).astype(bool)
        covariance_ = dispersion * np.linalg.inv(unscaled_precision_[keep][:,keep])

        SE = np.sqrt(np.diag(covariance_)) 
        index = self.feature_names_in_
        if self.fit_intercept:
            coef = np.hstack([self.intercept_, self.coef_])
            T = np.hstack([self.intercept_ / SE[0], self.coef_ / SE[1:]])
            index = ['intercept'] + index
        else:
            coef = self.coef_
            T = self.coef_ / SE

        summary_ = pd.DataFrame({'coef':coef,
                                 'std err': SE,
                                 'z': T,
                                 'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                index=index)
        return covariance_, summary_


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
    """
    fit_intercept: Literal[False] = False

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _finalize_family(self,
                         response):
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id,
                             strata_id=self.family.strata_id)

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
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _finalize_family(self,
                         response):
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id,
                             strata_id=self.family.strata_id)
    
    def get_LM(self):
        return CoxLM(family=self.family,
                     offset_id=self.offset_id,
                     weight_id=self.weight_id,
                     response_id=self.response_id)

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
        if prediction_type not in ['link', 'response']:
            raise ValueError("prediction_type should be one of 'link' or 'response' for Cox models")

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
                        event_data=event_split)

        split_w = sample_weight[split]
        dev_split = fam_split._coxdev(predictions[split], split_w).deviance
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
                       event_data=event_data)
        dev_full = fam_full._coxdev(predictions, sample_weight).deviance

        # now compute deviance on complement
        
        split_c = np.ones_like(predictions, bool)
        split_c[split] = 0

        event_c = event_data.iloc[split_c] # XXX presumes dataframe, could be ndarray
        fam_split_c = cls(tie_breaking=coxfam.tie_breaking,
                          event_id=coxfam.event_id,
                          status_id=coxfam.status_id,
                          start_id=coxfam.start_id,
                          event_data=event_c)
        split_c_w = sample_weight[split_c]
        dev_c = fam_split_c._coxdev(predictions[split_c], split_c_w).deviance

        w_sum = sample_weight.sum() - split_c_w.sum()
        return (dev_full - dev_c) / w_sum, w_sum


    




  
@dataclass(frozen=True)
class CoxCIndexScorer(Scorer):
    """
    Harrell's C index (R's ``type.measure="C"``), computed per fold and
    averaged with fold weights equal to the summed sample weights, as in R's
    ``cv.coxnet``. Strata are ignored unless ``stratified=True``, matching R.

    Use ``CoxCIndexScorer.from_family(family)`` to take the column names from
    a `CoxFamily` or `CoxFamilySpec`.
    """

    name: str = 'C-index'
    maximize: bool = True
    use_full_data: bool = True
    event_id: str = 'event'
    status_id: str = 'status'
    start_id: Optional[str] = None
    strata_id: Optional[str] = None
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
        if self.start_id is not None:
            start = event_split[self.start_id]
        if self.stratified and self.strata_id is not None:
            strata = event_split[self.strata_id]
        split_w = np.asarray(sample_weight)[split]

        value = c_index(np.asarray(predictions)[split],
                        event_split[self.event_id],
                        event_split[self.status_id],
                        start=start,
                        strata=strata,
                        sample_weight=split_w)
        return value, split_w.sum()
