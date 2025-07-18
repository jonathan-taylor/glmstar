from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from functools import partial

import numpy as np
import pandas as pd

from scipy.stats import norm as normal_dbn

from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

from coxdev import StratifiedCoxDeviance # , CoxDeviance
    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)
from .scoring import Scorer
from .regularized_glm import RegGLM
from .glmnet import GLMNet
from ._utils import _get_data

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
    tie_breaking : {'breslow', 'efron'}, default='efron'
        Method for handling ties in survival times.
    event_id : str, optional, default='event'
        Column name for event times.
    status_id : str, optional, default='status'
        Column name for event status (0=censored, 1=event).
    start_id : str, optional, default=None
        Column name for start times (for start-stop data).
    """
    tie_breaking: Literal['breslow', 'efron'] = 'efron'
    event_id: Optional[str] = 'event'
    status_id: Optional[str] = 'status'
    start_id: Optional[str] = None

@dataclass
class CoxFamilySpec(object):
    """
    Cox family specification for survival analysis.
    
    Parameters
    ----------
    event_data : InitVar[np.ndarray]
        Survival data containing event times, status, and optionally start times.
    tie_breaking : {'breslow', 'efron'}, default='efron'
        Method for handling ties in survival times.
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
    tie_breaking: Literal['breslow', 'efron'] = 'efron'
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
            strata = np.asarray(event_data[self.strata_id])
        else:
            strata = np.zeros(n, dtype=int)
        self.strata = strata
        
        if self.start_id is not None:
            start = event_data[self.start_id]
            self._coxdev = StratifiedCoxDeviance(
                np.asarray(event, float),
                status,
                start=np.asarray(start, float),
                strata=strata,
                tie_breaking=self.tie_breaking
            )
        else:
            start = None
            self._coxdev = StratifiedCoxDeviance(
                np.asarray(event, float),
                status,
                start=None,
                strata=strata,
                tie_breaking=self.tie_breaking
            )


    # GLMFamilySpec API
    def link(self, mu):
        return mu
    
    def deviance(self, 
                 y,
                 mu,
                 sample_weight):

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
                                 sample_weight):

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
        return self._coxdev.information(state.link_parameter,
                                        sample_weight)

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
class CoxNet(GLMNet):
    """
    CoxNet: Cox Proportional Hazards Model with Elastic Net regularization.
    
    Fits a Cox proportional hazards model with regularization along a path of lambda values.
    Supports both right-censored and start-stop survival data with Breslow or Efron tie-breaking.
    
    Parameters
    ----------
    fit_intercept : Literal[False], default=False
        Whether to fit an intercept. For Cox models, this is always False
        as the intercept is absorbed into the baseline hazard.
    regularized_estimator : BaseEstimator, default=RegCoxLM
        The regularized estimator class to use for fitting.
    """
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
                prediction_type='response',
                interpolation_grid=None):
        """
        Predict using the fitted CoxNet model.

        Parameters
        ----------
        X : Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized. If it is not a sparse matrix, a copy is made and
            standardized.
        prediction_type : str, default='response'
            Type of prediction to return. For Cox models, this is always the
            linear predictor (risk score), so this parameter is ignored.
        interpolation_grid : np.ndarray, optional
            Grid of lambda values for interpolation. If provided, coefficients are 
            interpolated to these values before prediction.

        Returns
        -------
        np.ndarray
            Predictions for each lambda value. Shape is (n_samples, n_lambdas)
            where n_lambdas is the number of lambda values in the fitted path
            or the length of interpolation_grid if provided.
        """
        if interpolation_grid is not None:
            grid_ = np.asarray(interpolation_grid)
            coefs_, intercepts_ = self.interpolate_coefs(grid_)
        else:
            grid_ = None
            coefs_, intercepts_ = self.coefs_, self.intercepts_

        intercepts_ = np.atleast_1d(intercepts_)
        coefs_ = np.atleast_2d(coefs_)
        linear_pred_ = coefs_ @ X.T + intercepts_[:, None]
        linear_pred_ = linear_pred_.T

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


    




  
