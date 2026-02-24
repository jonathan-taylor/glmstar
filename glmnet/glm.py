from typing import Union
from dataclasses import (dataclass,
                         asdict,
                         field)
from functools import partial
import logging
import warnings
   
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd

import scipy.sparse
from scipy.sparse.linalg import LinearOperator
from scipy.stats import norm as normal_dbn
from scipy.stats import t as t_dbn

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import LabelEncoder

from ._utils import (_parent_dataclass_from_child,
                     _get_data)

from .base import (Design,
                   _get_design)

from .irls import IRLS
from .family import (GLMFamilySpec,
                     BinomFamilySpec,
                     GLMState)


@dataclass
class GLMControl(object):
    """
    Control parameters for GLM fitting.
    
    Parameters
    ----------
    mxitnr: int
        Maximum number of quasi Newton iterations.
    epsnr: float
        Tolerance for quasi Newton iterations.
    big: float
        A large float, effectively `np.inf`.
    logging: bool
        Write info and debug messages to log?
    """
    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35
    logging: bool = False


@dataclass
class GLMBaseSpec(object):
    """
    Base specification for GLM models.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    control: GLMControl
        Parameters to control the solver.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    """
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    fit_intercept: bool = True
    standardize: bool = False
    control: GLMControl = field(default_factory=GLMControl)
    offset_id: Union[str,int] = None
    weight_id: Union[str,int] = None
    response_id: Union[str,int] = None
    exclude: list = field(default_factory=list)


@dataclass
class GLMResult(object):
    """
    Result object for GLM fitting.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    offset: bool
        Whether offset was used in fitting.
    converged: bool
        Did the algorithm converge?
    boundary: bool
        Was backtracking required due to getting near boundary of valid mean / natural parameters.
    obj_function: float
        Value of objective function (deviance + penalty).
    """
    family: GLMFamilySpec
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float


@dataclass
class GLMState(object):
    """
    State object for GLM fitting.
    
    Parameters
    ----------
    coef: np.ndarray
        Coefficient vector.
    intercept: np.ndarray
        Intercept value.
    obj_val: float
        Current objective value.
    pmin: float
        Minimum probability for binomial family.
    """
    coef: np.ndarray
    intercept: np.ndarray
    obj_val: float = np.inf
    pmin: float = 1e-9
    
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
        design: Design
            Design matrix.
        family: GLMFamilySpec
            GLM family specification.
        offset: np.ndarray, optional
            Offset vector.
        objective: callable, optional
            Objective function to evaluate.
        """
        _family = family.base
        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = _family.link.inverse(self.link_parameter)

        # shorthand
        self.mu = self.mean_parameter 
        self.eta = self.linear_predictor 

        if family.is_binomial: # isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = _family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):
        """
        Compute the score (gradient of log-likelihood).
        
        Parameters
        ----------
        family: GLMFamilySpec
            GLM family specification.
        y: np.ndarray
            Response variable.
        sample_weight: np.ndarray
            Sample weights.
            
        Returns
        -------
        np.ndarray
            Score vector.
        """
        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.link_parameter)
        
        # compute working residual
        y = np.asarray(y).reshape(-1)
        r = (y - self.mu) 
        return sample_weight * r * dmu_deta / varmu 


def compute_grad(glm_obj,
                 intercept,
                 coef,
                 design,
                 response,
                 offset=None,
                 scaled_input=False,
                 scaled_output=False,
                 sample_weight=None,
                 norm_weights=False):
    """
    Compute gradient for GLM fitting.
    
    Parameters
    ----------
    glm_obj: GLMBase
        GLM object.
    intercept: float
        Intercept value.
    coef: np.ndarray
        Coefficient vector.
    design: Design
        Design matrix.
    response: np.ndarray
        Response variable.
    offset: np.ndarray, optional
        Offset vector.
    scaled_input: bool
        Whether input is already scaled.
    scaled_output: bool
        Whether to return scaled output.
    sample_weight: np.ndarray, optional
        Sample weights.
    norm_weights: bool
        Whether to normalize weights.
        
    Returns
    -------
    tuple
        (score, saturated_score)
    """
    family = glm_obj._family

    if sample_weight is None:
        sample_weight = np.ones(design.shape[0])
        
    if not scaled_input:
        raw_state = GLMState(intercept=intercept,
                             coef=coef)
        scaled_state = design.raw_to_scaled(raw_state)
    else:
        scaled_state = GLMState(intercept=intercept,
                                coef=coef)

    scaled_state.update(design,
                        family,
                        offset=offset)

    saturated_score = scaled_state.logl_score(family,
                                              response, 
                                              sample_weight)
    scaled_score = design.T @ saturated_score
    if not scaled_output:
        score = design.scaler_.T @ scaled_score
    else:
        score = scaled_score
        
    if norm_weights:
        w_sum = sample_weight.sum()
        score /= w_sum
        saturated_score /= w_sum

    return score, saturated_score


@dataclass
class GLMRegularizer(object):
    """
    Regularizer for GLM fitting.
    
    Parameters
    ----------
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    warm_state: dict
        Warm start state.
    ridge_coef: float
        Ridge coefficient for a GLM. Added to objective **after** having divided by the sum of the weights.
    """
    fit_intercept: bool = False
    standardize: bool = False
    warm_state: dict = field(default_factory=dict)
    ridge_coef: float = 0

    def half_step(self,
                  state,
                  oldstate):
        """
        Compute half step between two states.
        
        Parameters
        ----------
        state: GLMState
            Current state.
        oldstate: GLMState
            Previous state.
            
        Returns
        -------
        GLMState
            Half step state.
        """
        klass = oldstate.__class__
        return klass(0.5 * (oldstate.coef + state.coef),
                     0.5 * (oldstate.intercept + state.intercept))

    def _debug_msg(self,
                   state):
        """Return debug message for state."""
        return f'Coef: {state.coef}, Intercept: {state.intercept}, Objective: {state.obj_val}'

    def check_state(self,
                    state):
        """
        Check state for validity.
        
        Parameters
        ----------
        state: GLMState
            State to check.
            
        Raises
        ------
        ValueError
            If state contains NaN values.
        """
        if np.any(np.isnan(state.coef)):
            raise ValueError('coef has NaNs')
        if np.isnan(state.intercept):
            raise ValueError('intercept is NaN')

    def newton_step(self,
                    design,
                    pseudo_response,
                    sample_weight,
                    cur_state):
        """
        Perform Newton step for GLM fitting.
        
        Parameters
        ----------
        design: Design
            Design matrix.
        pseudo_response: np.ndarray
            Pseudo response for IRLS.
        sample_weight: np.ndarray
            Sample weights.
        cur_state: GLMState
            Current state.
            
        Returns
        -------
        GLMState
            Updated state.
        """
        # ignored for GLM
        z = pseudo_response
        w = sample_weight

        D = np.ones(design.X.shape[1] + 1)
        D[0] = 0

        # this will produce coefficients on the raw scale

        if scipy.sparse.issparse(design.X):
            if self.standardize:
                raise ValueError('sklearn Ridge cannot fit without using raw standardized matrix')
            if self.ridge_coef == 0:
                lm = LinearRegression(fit_intercept=self.fit_intercept)
            else:
                lm = Ridge(fit_intercept=self.fit_intercept,
                           alpha=self.ridge_coef)
            lm.fit(design.X, z, sample_weight=w)
            coefnew = lm.coef_
            intnew = lm.intercept_

        else:
            sqrt_w = np.sqrt(w)

            if not self.standardize:
                XW = design.X * sqrt_w[:, None]
            else:
                X = design @ (np.identity(design.shape[1])[:,1:])
                XW = X * sqrt_w[:, None]
                
            if self.fit_intercept:
                D = np.diag(D)
                Wz = sqrt_w * z
                XW = np.concatenate([sqrt_w.reshape((-1,1)), XW], axis=1)
                Q = XW.T @ XW
                if self.ridge_coef != 0:
                    Q += self.ridge_coef * D
                V = XW.T @ Wz
                try:
                    beta = np.linalg.solve(Q, V)
                except LinAlgError as e:
                    if self.control.logging: logging.debug("Error in solve: possible singular matrix, trying pseudo-inverse")
                    if self.ridge_coef != 0:
                        XW = np.vstack([XW, np.sqrt(self.ridge_coef) * D])
                        Wz = np.hstack([Wz, np.zeros(Q.shape[0])])
                    beta = np.linalg.pinv(XW) @ Wz
                coefnew = beta[1:]
                intnew = beta[0]

            else:
                if self.ridge_coef != 0:
                    D = np.diag(D[1:])
                    XW = np.vstack([XW, np.sqrt(self.ridge_coef) * D])
                    z = np.hstack([z, np.zeros(D.shape[0])])
                    sqrt_w = np.hstack([sqrt_w, np.zeros(D.shape[0])])
                coefnew = np.linalg.pinv(XW) @ (sqrt_w * z)
                intnew = 0

        klass = cur_state.__class__
        self.warm_state = klass(coefnew,
                                intnew)
        
        return self.warm_state

    def objective(self, state):
        """
        Compute objective value.
        
        Parameters
        ----------
        state: GLMState
            Current state.
            
        Returns
        -------
        float
            Objective value (0 for GLM).
        """
        return 0


@dataclass
class GLMBase(BaseEstimator,
              GLMBaseSpec):
    """
    Base class for GLM models.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    control: GLMControl
        Parameters to control the solver.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    """

    def _get_regularizer(self,
                         nvars=None):
        """
        Get regularizer for fitting.
        
        Parameters
        ----------
        nvars: int, optional
            Number of variables.
            
        Returns
        -------
        GLMRegularizer
            Regularizer instance.
        """
        return GLMRegularizer(fit_intercept=self.fit_intercept)

    def _get_design(self,
                    X,
                    sample_weight):
        """
        Get design matrix.
        
        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        sample_weight: Optional[np.ndarray]
            Sample weights.
            
        Returns
        -------
        Design
            Design matrix.
        """
        return _get_design(X,
                           sample_weight,
                           standardize=self.standardize,
                           intercept=self.fit_intercept)

    def get_data_arrays(self, X, y, check=True):
        """
        Get data arrays for fitting.
        
        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        check: bool
            Run the `_check` method to validate `(X,y)`.
            
        Returns
        -------
        tuple
            (X, y, response, offset, weight)
        """
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check)
    
    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=None,
            check=True,
            fit_null=True):
        """
        Fit a GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: GLMRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer. For a GLM this is just the 0 function.
        warm_state: GLMState, optional
            Warm start state.
        dispersion: float, optional
            Dispersion parameter of GLM. If family is Gaussian, will be estimated as 
            minimized deviance divided by degrees of freedom.
        check: bool
            Run the `_check` method to validate `(X,y)`.
        fit_null: bool
            Whether to fit null model.
    
        Returns
        -------
        self: object
            GLM class instance.
        """
        nobs, nvar = X.shape
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        if not hasattr(self, "_family"):
            self._family = self._finalize_family(response=y)

        X, y, response, offset, weight = self.get_data_arrays(X, y, check=check)

        sample_weight = weight
        self.sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        response = np.asarray(response)

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        
        if not hasattr(self, "design_"):
            self.design_ = design = self._get_design(X,
                                                     normed_sample_weight)
        else:
            design = self.design_
            
        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(nvars=design.X.shape[1])
        self.regularizer_ = regularizer

        if warm_state is not None:
            self.regularizer_.warm_state = warm_state
            
        if fit_null or not hasattr(self.regularizer_, 'warm_state'):
            (null_state,
             self.null_deviance_) = self._family.get_null_deviance(
                                        response=response,
                                        sample_weight=sample_weight,
                                        offset=offset,
                                        fit_intercept=self.fit_intercept)


        if (hasattr(self.regularizer_, 'warm_state') and
            self.regularizer_.warm_state):
            state = self.regularizer_.warm_state
        else:
            state = self._family._get_null_state(null_state,
                                                 nvar)

        # for Cox, the state could have mu==eta so that this need not change
        def obj_function(response,
                         normed_sample_weight,
                         family,
                         regularizer,
                         state):
            val1 = family.deviance(response,
                                   state.mu,
                                   normed_sample_weight) / 2
            val2 = regularizer.objective(state)
            val = val1 + val2
            if self.control.logging: logging.debug(f'Computing objective, lambda: {regularizer.lambda_val}, alpha: {regularizer.alpha}, coef: {state.coef}, intercept: {state.intercept}, deviance: {val1}, penalty: {val2}')
            return val

        obj_function = partial(obj_function,
                               response.copy(),
                               normed_sample_weight.copy(),
                               self._family,
                               regularizer)
        
        state.update(design,
                     self._family,
                     offset,
                     obj_function)

        (converged,
         boundary,
         state,
         _) = IRLS(regularizer,
                   self._family,
                   design,
                   response,
                   offset,
                   normed_sample_weight,
                   state,
                   obj_function,
                   self.control)

        if self.summarize:
            self._information = self._family.information(state,
                                                         sample_weight)
            
        # checks on convergence and fitted values
        if not converged:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm did not converge")
        if boundary:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm stopped at boundary value")

        self.deviance_ = self._family.deviance(response,
                                               state.mean_parameter,
                                               sample_weight) # not the normalized weights!

        if offset is None:
            offset = np.zeros(y.shape[0])

        self._set_coef_intercept(state)

        if self._family.is_gaussian and dispersion is None:
            # usual estimate of sigma^2
            self.dispersion_ = self.deviance_ / (nobs - nvar - self.fit_intercept) 
            self.df_resid_ = nobs - nvar - self.fit_intercept
        else:
            self.dispersion_ = dispersion

        self.state_ = state
        return self
    
    def predict(self, X, prediction_type='response'):
        """
        Predict outcome of corresponding family.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        prediction_type: str
            One of "response" or "link". If "response" return a prediction on the mean scale,
            "link" on the link scale. Defaults to "response".

        Returns
        -------
        prediction: np.ndarray
            Predictions on the mean scale for family of a GLM.
        """
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        return self._family.predict(linpred, prediction_type=prediction_type)

    def score(self, X, y, sample_weight=None):
        """
        Compute weighted log-likelihood (i.e. negative deviance / 2) for test X and y using fitted model. Weights
        default to `np.ones_like(y) / y.shape[0]`.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.    

        Returns
        -------
        score: float
            Deviance of family for `(X, y, sample_weight)`.
        """
        mu = self.predict(X, prediction_type='response')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return -self._family.deviance(y, mu, sample_weight) / 2 # GLM specific

    def _set_coef_intercept(self, state):
        """
        Set coefficients and intercept from state.
        
        Parameters
        ----------
        state: GLMState
            State containing coefficients and intercept.
        """
        raw_state = self.design_.scaled_to_raw(state)
        self.coef_ = raw_state.coef
        self.intercept_ = raw_state.intercept


@dataclass
class GLM(GLMBase):
    """
    Generalized Linear Model.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    control: GLMControl
        Parameters to control the solver.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    summarize: bool
        Compute a Wald-type statistical summary from fitted GLM.
    ridge_coef: float
        Ridge coefficient for a GLM. Added to objective **after** having divided by the sum of the weights.
    """
    summarize: bool = False
    ridge_coef: float = 0

    def _get_regularizer(self,
                         nvars=None):
        """
        Get regularizer for fitting.
        
        Parameters
        ----------
        nvars: int, optional
            Number of variables.
            
        Returns
        -------
        GLMRegularizer
            Regularizer instance.
        """
        return GLMRegularizer(fit_intercept=self.fit_intercept,
                              ridge_coef=self.ridge_coef)

    def _finalize_family(self,
                         response):
        """
        Finalize family specification.
        
        Parameters
        ----------
        response: np.ndarray
            Response variable.
            
        Returns
        -------
        GLMFamilySpec
            Family specification.
        """
        if not hasattr(self, "_family"):
            return GLMFamilySpec.from_family(self.family, response)

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):
        """
        Fit a GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: GLMRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer. For a GLM this is just the 0 function.
        warm_state: GLMState, optional
            Warm start state.
        dispersion: float
            Dispersion parameter of GLM. If family is Gaussian, will be estimated as 
            minimized deviance divided by degrees of freedom.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        self: object
            GLM class instance.
        """
        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=dispersion,
                    check=check)

        weight = self.get_data_arrays(X, y, check=False)[-1]
            
        if self.summarize:
            self.covariance_, self.summary_ = self._summarize(self.exclude,
                                                              self.dispersion_,
                                                              weight,
                                                              X.shape)
        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   X_shape):
        """
        Compute summary statistics.
        
        Parameters
        ----------
        exclude: list
            Indices of variables to exclude.
        dispersion: float
            Dispersion parameter.
        sample_weight: np.ndarray
            Sample weights.
        X_shape: tuple
            Shape of design matrix.
            
        Returns
        -------
        tuple
            (covariance_, summary_)
        """
        if self.ridge_coef != 0:
            warnings.warn('Detected a non-zero ridge term: variance estimates are taken to be posterior variance estimates')

        # IRLS used normalized weights,
        # this unnormalizes them...

        unscaled_precision_ = self.design_.quadratic_form(self._information,
                                                          transformed=False)
        if self.ridge_coef != 0:
            D = np.ones(unscaled_precision_.shape[0])
            D[0] = 0
            unscaled_precision_ += self.ridge_coef * np.diag(D)
        
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

        if self._family.is_gaussian:
            n, p = X_shape
            self.df_resid_ = n - p - self.fit_intercept
            df_resid = self.df_resid_
        else:
            df_resid = np.inf

        if (df_resid < np.inf):
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     't': T,
                                     'P>|t|': 2 * t_dbn.sf(np.fabs(T),
                                                           df=df_resid)},
                                    index=index)
        else:
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     'z': T,
                                     'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                    index=index)
        return covariance_, summary_


@dataclass
class GaussianGLM(RegressorMixin, GLM):
    """
    Gaussian GLM for regression.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    control: GLMControl
        Parameters to control the solver.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    summarize: bool
        Compute a Wald-type statistical summary from fitted GLM.
    ridge_coef: float
        Ridge coefficient for a GLM. Added to objective **after** having divided by the sum of the weights.
    """
    pass


@dataclass
class BinomialGLM(ClassifierMixin, GLM):
    """
    Binomial GLM for classification.
    
    Parameters
    ----------
    family: BinomFamilySpec
        Specification of binomial family.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is False.
    control: GLMControl
        Parameters to control the solver.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    summarize: bool
        Compute a Wald-type statistical summary from fitted GLM.
    ridge_coef: float
        Ridge coefficient for a GLM. Added to objective **after** having divided by the sum of the weights.
    """
    family: BinomFamilySpec = field(default_factory=BinomFamilySpec)

    def get_data_arrays(self, X, y, check=True):
        """
        Get data arrays for fitting.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        tuple
            (X, y, labels, offset, weight)
        """
        X, y, response, offset, weight = super().get_data_arrays(X, y, check=check)
        encoder = LabelEncoder()
        labels = np.asfortranarray(encoder.fit_transform(response))
        self.classes_ = encoder.classes_
        if len(encoder.classes_) > 2:
            raise ValueError("BinomialGLM expecting a binary classification problem.")
        return X, y, labels, offset, weight

    def _finalize_family(self,
                         response):
        """
        Finalize family specification.
        
        Parameters
        ----------
        response: np.ndarray
            Response variable.
            
        Returns
        -------
        GLMFamilySpec
            Family specification.
        """
        if not hasattr(self, "_family"):
            _family = GLMFamilySpec.from_family(self.family, response)
            if not _family.is_binomial:
                msg = f'{self.__class__.__name__} expects a Binomial family.'
                warnings.warn(msg)
                if self.control.logging: logging.warn(msg)
            return _family
        
    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):
        """
        Fit a Binomial GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: GLMRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer. For a GLM this is just the 0 function.
        warm_state: GLMState, optional
            Warm start state.
        dispersion: float
            Dispersion parameter of GLM. If family is Gaussian, will be estimated as 
            minimized deviance divided by degrees of freedom.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        self: object
            BinomialGLM class instance.
        """
        self._family = self._finalize_family(response=y)

        return super().fit(X,
                           y,
                           sample_weight=sample_weight,
                           regularizer=regularizer,             # last 4 options non sklearn API
                           warm_state=warm_state,
                           dispersion=dispersion,
                           check=check)

    def predict(self, X, prediction_type='class'):
        """
        Predict outcome of corresponding family.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        prediction_type: str
            One of "response", "link" or "class". If "response" return a prediction on the mean scale,
            "link" on the link scale, and "class" as a class label. Defaults to "class".

        Returns
        -------
        prediction: np.ndarray
            Predictions on the mean scale for family of a GLM.
        """
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        pred = self._family.predict(linpred,
                                    prediction_type=prediction_type)
        if prediction_type == 'class':
            pred = self._classes[pred]
        return pred

    def predict_proba(self, X):
        """
        Probability estimates for a BinomialGLM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
 
        """
        prob_1 = self.predict(X, prediction_type='response')
        result = np.empty((prob_1.shape[0], 2))
        result[:,1] = prob_1
        result[:,0] = 1 - prob_1

        return result

