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
from scipy.stats import chi2 as chi2_dbn

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import LabelEncoder

from ._utils import (_parent_dataclass_from_child,
                     _get_data)

from .base import (Design,
                   DiagonalOperator,
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


def _state_on_design(state, design):
    """
    `state`, with coefficients and intercept on the original scale of the
    features, in the coordinates of `design` (standardized, if it is).
    """
    coef = np.asarray(state.coef)
    return state.__class__(coef=coef * design.scaling_,
                           intercept=state.intercept + (coef * design.centers_).sum())


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
    exclude: list = field(default_factory=list)

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

        # excluded variables are left out of the fit: their coefficients stay 0
        p = design.X.shape[1]
        keep = np.ones(p, bool)
        keep[np.asarray(self.exclude, int)] = False
        kept = np.nonzero(keep)[0]

        D = np.ones(kept.shape[0] + 1)
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
            lm.fit(design.X[:, kept], z, sample_weight=w)
            coefnew = np.zeros(p)
            coefnew[kept] = lm.coef_
            # the state is in the design's (centered) coordinates, as in
            # the dense case; design.X is the raw (uncentered) sparse matrix
            intnew = lm.intercept_ + coefnew @ design.centers_

        else:
            sqrt_w = np.sqrt(w)

            if not self.standardize:
                XW = design.X * sqrt_w[:, None]
            else:
                X = design @ (np.identity(design.shape[1])[:,1:])
                XW = X * sqrt_w[:, None]
            XW = XW[:, kept]

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
                coefnew = np.zeros(p)
                coefnew[kept] = beta[1:]
                intnew = beta[0]

            else:
                if self.ridge_coef != 0:
                    D = np.diag(D[1:])
                    XW = np.vstack([XW, np.sqrt(self.ridge_coef) * D])
                    z = np.hstack([z, np.zeros(D.shape[0])])
                    sqrt_w = np.hstack([sqrt_w, np.zeros(D.shape[0])])
                coefnew = np.zeros(p)
                coefnew[kept] = np.linalg.pinv(XW) @ (sqrt_w * z)
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
        y: Union[np.ndarray, pd.DataFrame]
            Target variables. It is highly recommended to pass a `pandas.DataFrame` 
            when utilizing `offset_id`, `weight_id`, or `response_id` so that
            these vectors can be extracted by column name robustly.
        check: bool
            Run the `sklearn.utils.check_X_y` method to validate `(X, response)`.
            
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
            Warm start: coefficients and intercept on the original scale of
            the features, as `coef_` and `intercept_`.
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
        # the family always sees the normalized weights, so that it gets one weight
        # vector (Cox's deviance is rebuilt for each new one); the deviances and the
        # information, linear in the weights, are scaled back by their total
        weight_total = sample_weight.sum()
        self.sample_weight_ = normed_sample_weight = sample_weight / weight_total
        # for residuals and the log-likelihood
        self._response, self._weight = response, weight
        
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
            self.regularizer_.warm_state = _state_on_design(warm_state, design)
            
        if fit_null or not hasattr(self.regularizer_, 'warm_state'):
            (null_state,
             null_deviance) = self._family.get_null_deviance(
                                        response=response,
                                        sample_weight=normed_sample_weight,
                                        offset=offset,
                                        fit_intercept=self.fit_intercept)
            self.null_deviance_ = weight_total * null_deviance


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
            # at the normalized weights: _summarize scales it by weight_total_
            self._information = self._family.information(state,
                                                         normed_sample_weight)
        self.weight_total_ = weight_total
            
        # checks on convergence and fitted values
        if not converged:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm did not converge")
        if boundary:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm stopped at boundary value")

        self.deviance_ = weight_total * self._family.deviance(response,
                                                              state.mean_parameter,
                                                              normed_sample_weight)

        if offset is None:
            offset = np.zeros(y.shape[0])

        self._set_coef_intercept(state)

        # the variables in the model, and the residual degrees of freedom
        self._kept = np.ones(nvar, bool)
        self._kept[np.asarray(self.exclude, int)] = False
        self.df_resid_ = int(np.sum(weight > 0)) - self._kept.sum() - self.fit_intercept

        if dispersion is None:
            # usual estimate of sigma^2 for the Gaussian, 1 otherwise
            dispersion = self.deviance_ / self.df_resid_ if self._family.is_gaussian else 1.
        self.dispersion_ = dispersion

        names = list(np.asarray(self.feature_names_in_)[self._kept])
        values = np.asarray(self.coef_)[self._kept]
        if self.fit_intercept:
            names, values = ['intercept'] + names, np.hstack([self.intercept_, values])
        self.params_ = pd.Series(values, index=names)

        self.state_ = state
        return self
    
    def predict(self, X, prediction_type='response', offset=None):
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
        offset: np.ndarray, optional
            Offset added to the linear predictor.

        Returns
        -------
        prediction: np.ndarray
            Predictions on the mean scale for family of a GLM.
        """
        return self._family.predict(self._linear_predictor(X, offset),
                                    prediction_type=prediction_type)

    def _linear_predictor(self, X, offset=None):
        X = self._drop_intercept_column(X)
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        linpred = np.asarray(linpred, float).reshape(-1)
        return linpred if offset is None else linpred + np.asarray(offset, float)

    def _drop_intercept_column(self, X, fitting=False):
        """
        A DataFrame `X` from a design builder such as ISLP's `ModelSpec` may have a
        constant column `intercept`, which duplicates the fitted intercept: it is
        dropped, with a warning when fitting.
        """
        if fitting:
            self._intercept_column = (self.fit_intercept and
                                      isinstance(X, pd.DataFrame) and
                                      'intercept' in X.columns and
                                      np.all(np.asarray(X['intercept']) == 1))
            if self._intercept_column:
                warnings.warn("dropping the constant column 'intercept' of X: "
                              "the model has an intercept (fit_intercept=True)")
        if (getattr(self, '_intercept_column', False) and
            isinstance(X, pd.DataFrame) and 'intercept' in X.columns):
            X = X.drop(columns='intercept')
        return X

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
                              ridge_coef=self.ridge_coef,
                              exclude=self.exclude)

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

    # the constant column 'intercept' of a ModelSpec design is dropped (see
    # `_drop_intercept_column`); RegGLM, fitted by GLMNet, does not
    _drops_intercept_column = True

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=None,
            check=True):
        """
        Fit a GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized. A DataFrame's constant column `intercept` (as made by
            ISLP's `ModelSpec`) is dropped when `fit_intercept` is True.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: GLMRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer. For a GLM this is just the 0 function.
        warm_state: GLMState, optional
            Warm start: coefficients and intercept on the original scale of
            the features, as `coef_` and `intercept_`.
        dispersion: float, optional
            Dispersion parameter of GLM. By default it is estimated as the
            deviance divided by the residual degrees of freedom for the Gaussian,
            and 1 for the other families.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        self: object
            GLM class instance.
        """
        if self._drops_intercept_column:
            X = self._drop_intercept_column(X, fitting=True)
        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=dispersion,
                    check=check)

        if self.summarize:
            self.covariance_, self.summary_ = self._summarize()
        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self):
        """
        The covariance of `params_` and the Wald summary table.

        Returns
        -------
        tuple
            (covariance_, summary_)
        """
        if self.ridge_coef != 0:
            warnings.warn('Detected a non-zero ridge term: variance estimates are taken to be posterior variance estimates')

        # the information is at the normalized weights: scale it back
        unscaled_precision_ = self.weight_total_ * self.design_.quadratic_form(self._get_information(),
                                                                               transformed=False)
        if self.ridge_coef != 0:
            D = np.ones(unscaled_precision_.shape[0])
            D[0] = 0
            unscaled_precision_ += self.ridge_coef * np.diag(D)
        
        keep = np.hstack([self.fit_intercept, self._kept]).astype(bool)
        covariance_ = self.dispersion_ * np.linalg.inv(unscaled_precision_[keep][:,keep])

        SE = np.sqrt(np.diag(covariance_)) 
        coef = self.params_.values
        T = coef / SE

        if self._family.is_gaussian:
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     't': T,
                                     'P>|t|': 2 * t_dbn.sf(np.fabs(T),
                                                           df=self.df_resid_)},
                                    index=self.params_.index)
        else:
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     'z': T,
                                     'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                    index=self.params_.index)
        return covariance_, summary_

    def _get_information(self):
        """The information of the fit (at the normalized weights)."""
        if not hasattr(self, '_information'):
            self._information = self._family.information(self.state_,
                                                         self.sample_weight_)
        return self._information

    def _get_covariance(self):
        if getattr(self, 'covariance_', None) is None:
            return self._summarize()[0]
        return self.covariance_

    def _quantile(self, alpha):
        # t for the Gaussian, whose dispersion is estimated, as in summary_
        if self._family.is_gaussian:
            return t_dbn.ppf(1 - alpha / 2, df=self.df_resid_)
        return normal_dbn.ppf(1 - alpha / 2)

    def conf_int(self, alpha=0.05):
        """
        Wald confidence intervals for `params_`, of level `1 - alpha`.

        Returns
        -------
        pd.DataFrame
            Columns `lower` and `upper`, indexed as `params_`.
        """
        se = np.sqrt(np.diag(self._get_covariance()))
        q = self._quantile(alpha)
        return pd.DataFrame({'lower': self.params_ - q * se,
                             'upper': self.params_ + q * se})

    def summary(self, alpha=0.05):
        """
        A statsmodels `Summary`: `tables[0]` describes the fit, `tables[1]` is
        `summary_` with the confidence intervals of `conf_int(alpha)`, as for
        a statsmodels results instance (so that ISLP's `summarize` works).
        """
        from statsmodels.iolib.summary import Summary
        from statsmodels.iolib.table import SimpleTable

        coefs = self._summarize()[1] if self.summary_ is None else self.summary_
        intervals = self.conf_int(alpha)
        # statsmodels' formats
        data = [[f'{row[0]:.4f}'] + [f'{v:.3f}' for v in row[1:]]
                for row in coefs.values]
        data = [row + [f'{lo:.3f}', f'{hi:.3f}']
                for row, lo, hi in zip(data, intervals['lower'], intervals['upper'])]
        headers = list(coefs.columns) + [f'[{alpha / 2:g}', f'{1 - alpha / 2:g}]']
        fit = [['Family:', getattr(self._family, 'name', None) or type(self._family.base).__name__],
               ['No. Observations:', f'{len(self._weight)}'],
               ['Df Residuals:', f'{self.df_resid_}'],
               ['Deviance:', f'{self.deviance_:.4f}'],
               ['Dispersion:', f'{self.dispersion_:.4g}'],
               ['Log-Likelihood:', f'{self.loglike_:.4f}'],
               ['AIC:', f'{self.aic_:.4f}']]
        smry = Summary()
        smry.tables = [SimpleTable(fit, title=f'{type(self).__name__} results'),
                       SimpleTable(data, headers=headers, stubs=list(coefs.index))]
        return smry

    @property
    def loglike_(self):
        """
        The log-likelihood, with the weights as prior weights (as R's `glm`), or
        the partial log-likelihood for Cox. The Gaussian's is at the maximum
        likelihood estimate of its variance, deviance / (number of observations).
        """
        if self._family.is_gaussian:
            scale = self.deviance_ / np.sum(self._weight > 0)
        else:
            scale = self.dispersion_
        return self._family.loglike(self._response, self.state_.mu, self._weight, scale)

    @property
    def aic_(self):
        """`-2 loglike_ + 2 k`, with `k` the number of coefficients (as statsmodels' GLM)."""
        return -2 * self.loglike_ + 2 * len(self.params_)

    @property
    def bic_(self):
        """`-2 loglike_ + log(n) k`, with `k` the number of coefficients."""
        return -2 * self.loglike_ + np.log(np.sum(self._weight > 0)) * len(self.params_)

    @property
    def fittedvalues_(self):
        """The fitted values, on the scale of `predict(prediction_type='response')`."""
        return self._family.predict(self.state_.link_parameter)

    @property
    def resid_(self):
        """The default residuals of the family: see `residuals`."""
        return self.residuals()

    def residuals(self, kind=None):
        """
        Residuals of the fit.

        Parameters
        ----------
        kind: str, optional
            For a GLM, 'response' (the default, y - mu), 'pearson' or 'deviance',
            with the weights as prior weights. For Cox, 'martingale' (the default)
            or 'deviance'.
        """
        kwargs = {} if kind is None else {'kind': kind}
        return self._family.residuals(self._response, self.state_.mu, self._weight, **kwargs)

    def _rows(self, X):
        """[1, X] on the variables in the model, as `params_`, keeping a sparse X sparse."""
        X = self._drop_intercept_column(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not scipy.sparse.issparse(X):
            X = np.asarray(X, float)
        else:
            X = scipy.sparse.csc_array(X)
        return X[:, np.nonzero(self._kept)[0]]

    def _quadratic_rows(self, X, C):
        """The diagonal of [1, X] C [1, X]^T, with the intercept's row only if fitted."""
        X = self._rows(X)
        if self.fit_intercept:
            C00, c, C = C[0, 0], C[0, 1:], C[1:, 1:]
        XC = np.asarray(X @ C)
        if scipy.sparse.issparse(X):
            q = np.asarray(X.multiply(XC).sum(1)).reshape(-1)
        else:
            q = (X * XC).sum(1)
        if self.fit_intercept:
            q = q + 2 * np.asarray(X @ c).reshape(-1) + C00
        return q

    def predict_se(self, X):
        """
        Standard errors of the linear predictor at the rows of `X`: the square root
        of the diagonal of X Cov X^T, with Cov the covariance of `params_`.
        """
        return np.sqrt(self._quadratic_rows(X, self._get_covariance()))

    def predict_conf_int(self, X, alpha=0.05, prediction_type='response', offset=None):
        """
        Confidence intervals of level `1 - alpha` for the linear predictor at the
        rows of `X`, from `predict_se`, mapped to the scale of `prediction_type`.

        Returns
        -------
        pd.DataFrame
            Columns `fit`, `lower` and `upper`.
        """
        eta = self._linear_predictor(X, offset)
        half = self._quantile(alpha) * self.predict_se(X)
        fit, a, b = (self._family.predict(v, prediction_type=prediction_type)
                     for v in (eta, eta - half, eta + half))
        return pd.DataFrame({'fit': fit,
                             'lower': np.minimum(a, b),
                             'upper': np.maximum(a, b)})

    def leverage(self, X):
        """
        The leverage of each observation of the fit, the diagonal of
        X (X^T W X)^{-1} X^T W, with `X` the design the model was fitted to and W
        the information in the linear predictor. For a GLM, W is diagonal and this
        is the diagonal of the hat matrix. For Cox, W is not diagonal: this is the
        generalized leverage, which R does not report.
        """
        C = self._get_covariance() / self.dispersion_
        W = self._get_information()
        if isinstance(W, DiagonalOperator):
            return self.weight_total_ * W.weights * self._quadratic_rows(X, C)
        X = self._rows(X)
        X = X.toarray() if scipy.sparse.issparse(X) else X
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return self.weight_total_ * ((X @ C) * (W @ X)).sum(1)


def anova(*fits):
    """
    Analysis of deviance for nested fits to the same data, from the smallest model
    to the largest: R's `anova(..., test="Chisq")`, for every family. The
    statistic is the drop in deviance divided by the dispersion of the largest
    model, compared with chi-squared on the drop in residual degrees of freedom.

    Returns
    -------
    pd.DataFrame
        Columns `df_resid`, `deviance`, `df_diff`, `dev_diff`, `chi2` and `P(>chi2)`.
    """
    if len({len(fit._weight) for fit in fits}) > 1:
        raise ValueError('the fits are to different numbers of observations')
    scale = min(fits, key=lambda fit: fit.df_resid_).dispersion_
    df = np.array([fit.df_resid_ for fit in fits], float)
    dev = np.array([fit.deviance_ for fit in fits], float)
    df_diff = np.hstack([np.nan, -np.diff(df)])
    dev_diff = np.hstack([np.nan, -np.diff(dev)])
    chi2 = np.abs(dev_diff) / scale
    return pd.DataFrame({'df_resid': df,
                         'deviance': dev,
                         'df_diff': df_diff,
                         'dev_diff': dev_diff,
                         'chi2': chi2,
                         'P(>chi2)': chi2_dbn.sf(chi2, np.abs(df_diff))})


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
        """Prepare and validate data arrays for binomial regression.

        For binomial regression, the response can be specified as a 1D array of labels,
        or as a 2D array of shape (n_samples, 2) containing pairs of (trials, successes).
        If provided as (trials, successes), `sample_weight` will be multiplied by the 
        number of trials, and the response will be transformed into proportions. This
        assumption is documented because users might mistakenly use trials as weights
        without accounting for them in the data shape.

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
        
        if response.ndim == 2 and response.shape[1] == 2:
            trials = response[:, 0]
            successes = response[:, 1]
            if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
                raise ValueError("For (trials, successes) input, trials must be > 0 and 0 <= successes <= trials.")
            
            weight = weight * trials
            labels = successes / trials
            self.classes_ = np.array([0, 1])
            if hasattr(self, '_family'):
                self._family.classes_ = self.classes_
        else:
            if response.ndim == 2 and response.shape[1] == 1:
                response = response.ravel()
            encoder = LabelEncoder()
            labels = np.asfortranarray(encoder.fit_transform(response))
            self.classes_ = encoder.classes_
            if hasattr(self, '_family'):
                self._family.classes_ = self.classes_
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
            Warm start: coefficients and intercept on the original scale of
            the features, as `coef_` and `intercept_`.
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

    def predict(self, X, prediction_type='class', offset=None):
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
        eta = self._linear_predictor(X, offset)
        if prediction_type == 'class':
            # the family may be a plain GLMFamilySpec, without classes
            prob = self._family.predict(eta, prediction_type='response')
            return self.classes_[(prob > 0.5).astype(int)]
        return self._family.predict(eta, prediction_type=prediction_type)

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

