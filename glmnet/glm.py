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
from scipy.special import gammaln, xlogy, xlog1py

from statsmodels.genmod.families import family as sm_family

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import _get_feature_names

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
        Tolerance for quasi Newton iterations: they stop when the change in
        the objective (the deviance over twice the sum of the weights)
        relative to `0.1 + |objective|` is below it. Default `1e-10`: as the
        objective is per observation this is about R's `glm.control(epsilon=1e-8)`;
        `1e-6` stopped early, e.g. near separation or for non-canonical
        links. (The penalized fits along a path, with `RegGLMControl`, keep
        glmnet's `1e-6`.)
    big: float
        A large float, effectively `np.inf`.
    logging: bool
        Write info and debug messages to log?
    """
    mxitnr: int = 25
    epsnr: float = 1e-10
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
            coefnew = np.zeros(p)
            if kept.shape[0] > 0:
                lm.fit(design.X[:, kept], z, sample_weight=w)
                coefnew[kept] = lm.coef_
                intercept = lm.intercept_
            else:
                # the intercept alone
                intercept = np.sum(w * z) / np.sum(w) if self.fit_intercept else 0.
            # the state is in the design's (centered and scaled) coordinates, as
            # in the dense case; design.X is the raw sparse matrix
            intnew = intercept + coefnew @ design.centers_
            coefnew = coefnew * design.scaling_

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
                    logging.debug("Error in solve: possible singular matrix, trying pseudo-inverse")
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
            Dispersion parameter of GLM. By default it is estimated, by
            Pearson's chi-squared over the residual degrees of freedom, for the
            families with a free dispersion (Gaussian, Gamma, inverse Gaussian,
            Tweedie...), and is 1 for the binomial, Poisson and negative binomial.
        check: bool
            Run the `_check` method to validate `(X,y)`.
        fit_null: bool
            Whether to fit null model.
    
        Returns
        -------
        self: object
            GLM class instance.
        """
        if not getattr(self, '_path_refit', False):
            # a fit starts afresh: the design, the family (Cox's holds the
            # event data) and the information are of these data, not of an
            # earlier fit; GLMNet refits its RegGLM to the same data along
            # the path, keeping them (_path_refit)
            for attr in ['design_', '_family', '_information']:
                self.__dict__.pop(attr, None)

        nobs, nvar = X.shape

        # the labels of the observations of a DataFrame X, for residuals and
        # leverage, and its column names, checked at predict as sklearn does
        self._train_index = X.index if isinstance(X, pd.DataFrame) else None
        self._fit_feature_names = _get_feature_names(X)

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
            # as R's glm.fit; RegGLM's fits along GLMNet's path do not warn
            if self._unpenalized:
                warnings.warn(f'{self.__class__.__name__}: the fit did not converge in '
                              f'mxitnr={self.control.mxitnr} iterations (epsnr='
                              f'{self.control.epsnr}); increase control.mxitnr')
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
        self._kept[np.asarray(self._excluded(), int)] = False
        self.df_resid_ = int(np.sum(weight > 0)) - self._kept.sum() - self.fit_intercept

        # the dispersion is estimated, as R's summary.glm and statsmodels do, by
        # Pearson's chi-squared over the residual degrees of freedom when the
        # family has a free dispersion (Gaussian with any link, Gamma, inverse
        # Gaussian, Tweedie...) and none is given; it is 1 for the binomial,
        # Poisson and negative binomial, and for Cox
        self._estimated_dispersion = dispersion is None and not self._fixed_dispersion()
        if self._estimated_dispersion:
            dispersion = (np.sum(self.residuals('pearson', state=state)**2) / self.df_resid_
                          if self.df_resid_ > 0 else np.nan)
        elif dispersion is None:
            dispersion = 1.
        self.dispersion_ = dispersion

        names = list(np.asarray(self.feature_names_in_)[self._kept])
        values = np.asarray(self.coef_)[self._kept]
        if self.fit_intercept:
            names, values = ['intercept'] + names, np.hstack([self.intercept_, values])
        self.params_ = pd.Series(values, index=names)
        self.feature_names_ = np.asarray(self.params_.index)

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
            Predictions of shape `(n_samples,)` on the scale of `prediction_type`.
            An array even if `X` is a DataFrame, as for sklearn's estimators: for
            a Series on the index of `X`, use `pd.Series(fit.predict(X), index=X.index)`.
        """
        return self._family.predict(self._linear_predictor(X, offset),
                                    prediction_type=prediction_type)

    def _linear_predictor(self, X, offset=None):
        X = self._drop_intercept_column(X)
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        linpred = np.asarray(linpred, float).reshape(-1)
        return linpred if offset is None else linpred + np.asarray(offset, float)

    def _excluded(self):
        """`exclude`, and the constant columns of X excluded by `_drop_intercept_column`."""
        excluded = list(self.exclude) + getattr(self, '_excluded_intercept', [])
        return sorted(set(excluded) | set(getattr(self, '_excluded_constant', [])))

    # a constant column of X is not identifiable with an intercept; for Cox
    # (whose partial likelihood is unchanged by a shift of the linear
    # predictor) it is not identifiable at all
    _shift_invariant = False

    def _constant_columns(self, X):
        """Indices of the constant columns of `X` (dense, sparse or DataFrame)."""
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csc_array(X)
            n = X.shape[0]
            nnz = np.diff(X.indptr)
            const = []
            for j in range(X.shape[1]):
                values = X.data[X.indptr[j]:X.indptr[j+1]]
                if nnz[j] == 0:
                    const.append(j)
                elif nnz[j] == n and np.all(values == values[0]):
                    const.append(j)
                elif nnz[j] < n and np.all(values == 0):
                    const.append(j)
            return const
        X = np.asarray(X)
        if X.dtype == object or X.shape[0] == 0:
            return []
        return [int(j) for j in np.nonzero(np.all(X == X[:1], axis=0))[0]]

    def _drop_intercept_column(self, X, fitting=False):
        """
        A DataFrame `X` from a design builder such as ISLP's `ModelSpec` may have a
        constant column `intercept`, which duplicates the fitted intercept: it is
        dropped, without a warning as this is ISLP's convention (a warning would
        be raised at every fold of a cross-validation or step of a selection).
        As the only column (the intercept-only model) it is kept but excluded
        instead.
        """
        if fitting:
            self._intercept_column = (self.fit_intercept and
                                      isinstance(X, pd.DataFrame) and
                                      'intercept' in X.columns and
                                      np.all(np.asarray(X['intercept']) == 1))
            self._excluded_intercept = []
            if self._intercept_column and X.shape[1] == 1:
                self._intercept_column = False
                self._excluded_intercept = [0]
        if (getattr(self, '_intercept_column', False) and
            isinstance(X, pd.DataFrame) and 'intercept' in X.columns):
            X = X.drop(columns='intercept')
        if fitting:
            # other constant columns (e.g. the column of ones of
            # np.power.outer(x, np.arange(d+1))) would make the fit singular:
            # they are excluded (their coefficients are 0), with a warning
            self._excluded_constant = []
            if ((self.fit_intercept or self._shift_invariant) and
                not self._excluded_intercept):
                const = self._constant_columns(X)
                if const:
                    names = (list(np.asarray(X.columns)[const]) if isinstance(X, pd.DataFrame)
                             else const)
                    reason = ('the model has an intercept (fit_intercept=True)' if self.fit_intercept
                              else 'they are not identifiable')
                    warnings.warn(f'excluding the constant column(s) {names} of X: {reason}')
                    self._excluded_constant = const
        if not fitting and self._unpenalized:
            self._check_feature_names_at_predict(X)
        return X

    _unpenalized = False

    def _check_feature_names_at_predict(self, X):
        """
        As sklearn's `_check_feature_names(X, reset=False)`: the column names of a
        DataFrame `X` must be those of the `X` of the fit (after dropping a
        ModelSpec `intercept` column), in the same order, or a ValueError is
        raised; a warning if only one of them had names.
        """
        fitted = getattr(self, '_fit_feature_names', None)
        names = _get_feature_names(X)
        klass = type(self).__name__
        if fitted is None and names is None:
            return
        if names is None:
            warnings.warn(f'X does not have valid feature names, but {klass} was fitted '
                          'with feature names')
            return
        if fitted is None:
            warnings.warn(f'X has feature names, but {klass} was fitted without feature names')
            return
        if len(fitted) == len(names) and np.all(fitted == names):
            return
        message = 'The feature names should match those that were passed during fit.\n'
        unseen = sorted(set(names) - set(fitted))
        missing = sorted(set(fitted) - set(names))
        if unseen:
            message += 'Feature names unseen at fit time:\n' + ''.join(f'- {n}\n' for n in unseen)
        if missing:
            message += ('Feature names seen at fit time, yet now missing:\n' +
                        ''.join(f'- {n}\n' for n in missing))
        if not unseen and not missing:
            message += 'Feature names must be in the same order as they were in fit.\n'
        raise ValueError(message)

    def score(self, X, y, sample_weight=None):
        """
        Compute weighted log-likelihood (i.e. negative deviance / 2) for test X and y using fitted model. Weights
        default to `np.ones_like(y)`, so this is a sum over the observations, not an
        average: it grows with the number of observations, as `CoxLM.score`.

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
            Minus half the deviance of family for `(X, y, sample_weight)`.
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
    sandwich: bool
        Use the sandwich (robust) estimate of the covariance of the
        coefficients, `covariance_`, in `summary_`, `predict_se` and
        `confidence_interval`, in place of the model-based one. Default is False.
    ridge_coef: float
        Ridge coefficient for a GLM. Added to objective **after** having divided by the sum of the weights.

    Attributes
    ----------
    params_: pd.Series
        The coefficients, with the intercept (if fitted) first.
    feature_names_: np.ndarray
        The names labelling the entries of `params_` and the rows and columns
        of `covariance_`: `'intercept'` if it is fitted, then the variables in
        the model (leaving out `exclude` and a dropped `intercept` column), i.e.
        `np.asarray(params_.index)`. Unlike sklearn's `feature_names_in_` (the
        columns of the `X` passed to `fit`), it includes the intercept and
        leaves out the excluded variables.
    covariance_: np.ndarray
        The covariance of `params_` (if `summarize`), labelled by `feature_names_`.
    """
    summarize: bool = False
    sandwich: bool = False
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
                              exclude=self._excluded())

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

    # the analysis of an unpenalized fit: the constant column 'intercept' of a
    # ModelSpec design is dropped (see `_drop_intercept_column`), and the
    # log-likelihood is computed; RegGLM, fitted along GLMNet's path, does neither
    _unpenalized = True

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
            Dispersion parameter of GLM. By default it is estimated, by
            Pearson's chi-squared over the residual degrees of freedom, for the
            families with a free dispersion (Gaussian, Gamma, inverse Gaussian,
            Tweedie...), and is 1 for the binomial, Poisson and negative binomial.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        self: object
            GLM class instance.
        """
        if self._unpenalized:
            X = self._drop_intercept_column(X, fitting=True)
        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=dispersion,
                    check=check)

        if self._unpenalized:
            # with k the number of coefficients and n the number of observations
            k, n = len(self.params_), np.sum(self._weight > 0)
            self.loglike_ = self._loglike()
            self.aic_ = -2 * self.loglike_ + 2 * k
            self.bic_ = -2 * self.loglike_ + np.log(n) * k

        if self.summarize:
            self.covariance_, self.summary_ = self._summarize()
        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self):
        """
        The covariance of `params_` and the Wald summary table, with the
        confidence intervals of level 0.95.

        Returns
        -------
        tuple
            (covariance_, summary_)
        """
        if self.ridge_coef != 0:
            warnings.warn('Detected a non-zero ridge term: variance estimates are taken to be posterior variance estimates')

        bread = np.linalg.inv(self._precision())
        if self.sandwich:
            covariance_ = bread @ self._score_meat() @ bread
        else:
            covariance_ = self.dispersion_ * bread

        SE = np.sqrt(np.diag(covariance_)) 
        coef = self.params_.values
        T = coef / SE

        if self._use_t():
            stat, pval = 't', 2 * t_dbn.sf(np.fabs(T), df=self.df_resid_)
        else:
            stat, pval = 'z', 2 * normal_dbn.sf(np.fabs(T))
        q = self._quantile(0.05)
        summary_ = pd.DataFrame({'coef': coef,
                                 'std err': SE,
                                 stat: T,
                                 f'P>|{stat}|': pval,
                                 '[0.025': coef - q * SE,
                                 '0.975]': coef + q * SE},
                                index=self.params_.index)
        return covariance_, summary_

    def _keep(self):
        """The intercept (if fitted) and the variables in the model."""
        return np.hstack([self.fit_intercept, self._kept]).astype(bool)

    def _precision(self):
        """
        The information of `params_` at the weights of the fit, without the
        dispersion, with any ridge term.
        """
        # the information is at the normalized weights: scale it back
        precision = self.weight_total_ * self.design_.quadratic_form(self._get_information(),
                                                                     transformed=False)
        if self.ridge_coef != 0:
            D = np.ones(precision.shape[0])
            D[0] = 0
            precision += self.ridge_coef * np.diag(D)
        keep = self._keep()
        return precision[keep][:, keep]

    def _score_meat(self):
        """
        The middle of the sandwich, the sum over the observations of the outer
        products of their scores of `params_`, without the dispersion.
        """
        # the score of observation i is x_i w_i (y_i - mu_i) (dmu/deta)_i / V(mu_i)
        r = self.state_.logl_score(self._family, self._response, self._weight)
        meat = self.design_.quadratic_form(DiagonalOperator(r**2), transformed=False)
        keep = self._keep()
        return meat[keep][:, keep]

    def _fixed_dispersion(self):
        """Is the dispersion of the family fixed at 1 (binomial, Poisson, negative binomial, Cox)?"""
        base = getattr(self._family, 'base', None)
        return base is None or isinstance(base, (sm_family.Binomial,
                                                 sm_family.Poisson,
                                                 sm_family.NegativeBinomial))

    def _use_t(self):
        """
        t on `df_resid_` (rather than z) for the Gaussian with identity link, whose
        dispersion is estimated, with the model-based covariance: the case where the
        Wald statistic is exactly t.
        """
        return (self._family.is_gaussian and
                getattr(self, '_estimated_dispersion', False) and
                not getattr(self, 'sandwich', False))

    def _loglike(self):
        y = np.asarray(self._response, float).reshape(-1)
        base = getattr(self._family, 'base', None)
        if isinstance(base, sm_family.Binomial):
            # y the proportion of w trials, as R's binomial()$aic: log choose(w, w y) +
            # w y log(mu) + w (1 - y) log(1 - mu); statsmodels' loglike takes one trial
            # per row, so it is wrong for counts. For 0/1 y the binomial coefficient
            # is 1 (0 for any weights), the Bernoulli log-likelihood times w
            w, mu = self._weight, self.state_.mu
            s = w * y
            return np.sum(gammaln(w + 1) - gammaln(s + 1) - gammaln(w - s + 1) +
                          xlogy(s, mu) + xlog1py(w - s, -mu))
        # the Gaussian's (any link) at the maximum likelihood estimate of its variance
        if isinstance(base, sm_family.Gaussian):
            scale = self.deviance_ / np.sum(self._weight > 0)
        else:
            scale = self.dispersion_
        return self._family.base.loglike(y,
                                         self.state_.mu,
                                         var_weights=self._weight,
                                         scale=scale)

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
        # t or z, as in summary_
        if self._use_t():
            return t_dbn.ppf(1 - alpha / 2, df=self.df_resid_)
        return normal_dbn.ppf(1 - alpha / 2)

    def summary(self, alpha=0.05):
        """
        A statsmodels `Summary`: `tables[0]` describes the fit, `tables[1]` is
        `summary_` with confidence intervals of level `1 - alpha`, as for a
        statsmodels results instance (so that ISLP's `summarize` works).
        """
        from statsmodels.iolib.summary import Summary
        from statsmodels.iolib.table import SimpleTable

        coefs = (self._summarize()[1] if self.summary_ is None else self.summary_).iloc[:, :4]
        q = self._quantile(alpha)
        lower, upper = coefs['coef'] - q * coefs['std err'], coefs['coef'] + q * coefs['std err']
        # statsmodels' formats
        data = [[f'{row[0]:.4f}'] + [f'{v:.3f}' for v in row[1:]] + [f'{lo:.3f}', f'{hi:.3f}']
                for row, lo, hi in zip(coefs.values, lower, upper)]
        headers = list(coefs.columns) + [f'[{alpha / 2:g}', f'{1 - alpha / 2:g}]']
        fit = [['Family:', getattr(self._family, 'name', None) or type(self._family.base).__name__],
               ['No. Observations:', f'{len(self._weight)}'],
               ['Df Residuals:', f'{self.df_resid_}'],
               ['Deviance:', f'{self.deviance_:.4f}'],
               ['Dispersion:', f'{self.dispersion_:.4g}'],
               ['Log-Likelihood:', f'{self.loglike_:.4f}'],
               ['AIC:', f'{self.aic_:.4f}'],
               ['Covariance:', 'sandwich' if self.sandwich else 'model']]
        smry = Summary()
        smry.tables = [SimpleTable(fit, title=f'{type(self).__name__} results'),
                       SimpleTable(data, headers=headers, stubs=list(coefs.index))]
        return smry

    def residuals(self, kind='response', state=None):
        """
        Residuals of the fit: 'response' (y - mu), 'pearson' or 'deviance', with
        the weights as prior weights, as R's `residuals.glm`.
        """
        y = np.asarray(self._response, float).reshape(-1)
        state = self.state_ if state is None else state
        mu, w = state.mu, self._weight
        family = self._family.base
        if kind == 'response':
            value = y - mu
        elif kind == 'pearson':
            value = np.sqrt(w) * (y - mu) / np.sqrt(family.variance(mu))
        elif kind == 'deviance':
            value = family.resid_dev(y, mu, var_weights=w)
        else:
            raise ValueError("residuals are 'response', 'pearson' or 'deviance'")
        return self._on_train_index(value, kind)

    def _on_train_index(self, value, name):
        """
        A vector over the observations of the fit: a Series on the index of X if
        it was a DataFrame, else an array.
        """
        value = np.asarray(value, float).reshape(-1)
        index = getattr(self, '_train_index', None)
        return value if index is None else pd.Series(value, index=index, name=name)

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

        Returns
        -------
        np.ndarray or pd.Series
            A Series on the index of `X` if it is a DataFrame, else an array.
        """
        return _on_rows(X, np.sqrt(self._quadratic_rows(X, self._get_covariance())), 'std err')

    def leverage(self, X):
        """
        The leverage of each observation of the fit, the diagonal of
        X (X^T W X)^{-1} X^T W, with `X` the design the model was fitted to and W
        the information in the linear predictor. For a GLM, W is diagonal and this
        is the diagonal of the hat matrix. For Cox, W is not diagonal: this is the
        generalized leverage, which R does not report.

        Returns
        -------
        np.ndarray or pd.Series
            A Series on the index of the fit's `X` if it was a DataFrame, else an
            array.
        """
        C = np.linalg.inv(self._precision())
        W = self._get_information()
        if isinstance(W, DiagonalOperator):
            value = self.weight_total_ * W.weights * self._quadratic_rows(X, C)
        else:
            X = self._rows(X)
            X = X.toarray() if scipy.sparse.issparse(X) else X
            if self.fit_intercept:
                X = np.column_stack([np.ones(X.shape[0]), X])
            value = self.weight_total_ * ((X @ C) * (W @ X)).sum(1)
        return self._on_train_index(value, 'leverage')


def _on_rows(X, value, name):
    """A vector at the rows of `X`: a Series on the index of a DataFrame `X`, else an array."""
    if not isinstance(X, pd.DataFrame):
        return value
    return pd.Series(value, index=X.index, name=name)


def _interval_frame(X, estimate, half):
    """
    Intervals at the rows of `X`: always a DataFrame (its columns are named),
    on the index of a DataFrame `X`, else on a RangeIndex.
    """
    index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(estimate))
    return pd.DataFrame({'estimate': estimate,
                         'lower': estimate - half,
                         'upper': estimate + half},
                        index=index)


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
    if any(fit.sandwich for fit in fits):
        warnings.warn('a fit uses the sandwich covariance: the drop in deviance is not '
                      'chi-squared when the model is wrong, so the p-values of anova are '
                      'not robust')
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


def confidence_interval(fit, X_new, alpha=0.05):
    """
    Wald confidence intervals of level `1 - alpha` for the linear
    combinations of `fit.params_` given by the rows of `X_new` (with the
    intercept if fitted), i.e. for the linear predictor at `X_new` without
    an offset, for an unpenalized `GLM` or `CoxLM`. The intervals for the
    coefficients themselves are in `fit.summary_`.

    Returns
    -------
    pd.DataFrame
        Columns `estimate`, `lower` and `upper`, on the index of `X_new` if it
        is a DataFrame, else on a `RangeIndex`.
    """
    estimate = fit._linear_predictor(X_new)
    half = fit._quantile(alpha) * np.asarray(fit.predict_se(X_new))
    return _interval_frame(X_new, estimate, half)


def prediction_interval(fit, X_new, alpha=0.05, weights=None):
    """
    Prediction intervals of level `1 - alpha` for new responses at the rows of
    `X_new`, for an unpenalized Gaussian `GLM` with identity link, an estimated
    dispersion and the model-based covariance (`sandwich=False`), as R's
    `predict.lm(interval="prediction")`: the linear predictor plus or minus the t
    quantile on `fit.df_resid_` times sqrt(se^2 + sigma^2 / w), with se from
    `fit.predict_se`, sigma^2 the estimated `fit.dispersion_` and `w` the prior
    weights of the new observations (default 1).

    Returns
    -------
    pd.DataFrame
        Columns `estimate`, `lower` and `upper`, on the index of `X_new` if it is a
        DataFrame, else on a `RangeIndex`.
    """
    if not (getattr(fit, '_unpenalized', False) and fit._use_t()):
        raise ValueError('prediction intervals are for an unpenalized Gaussian GLM with '
                         'identity link, an estimated dispersion and sandwich=False')
    estimate = fit._linear_predictor(X_new)
    w = np.ones(estimate.shape[0]) if weights is None else np.asarray(weights, float)
    half = fit._quantile(alpha) * np.sqrt(np.asarray(fit.predict_se(X_new))**2 + fit.dispersion_ / w)
    return _interval_frame(X_new, estimate, half)

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
            Dispersion parameter of GLM, 1 for the binomial.
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
            Predictions of shape `(n_samples,)`: labels from `classes_` if
            `prediction_type` is "class", else on the scale of `prediction_type`.
            An array even if `X` is a DataFrame, as for sklearn's estimators.
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

