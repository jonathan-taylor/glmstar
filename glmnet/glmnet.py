from copy import copy
import logging
import warnings
from itertools import product

from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.base import (BaseEstimator,
                          clone)
from sklearn.model_selection import (cross_val_predict,
                                     check_cv,
                                     KFold)
from sklearn.model_selection._validation import indexable

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y

from statsmodels.genmod.families import family as sm_family

from .regularized_glm import (RegGLMControl,
                              RegGLM)

from .glm import (GLM,
                  GLMState,
                  GLMFamilySpec)
from ._utils import _get_data, _check_offset
from .scorer import (PathScorer,
                     ScorePath,
                     RelaxedScorePath,
                     _tune_relaxed)



def _check_gamma(gamma):
    """
    Validate a single `gamma` for blending the lasso and relaxed fits; as
    R's `checkgamma.relax`, it must lie in [0, 1].
    """
    gamma = float(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError(f'gamma should be in [0, 1], got {gamma}')
    return gamma


class _RelaxedPredictor(BaseEstimator):
    """
    Fit a relaxed path estimator and predict for each value of `gamma`,
    stacked along axis 1; lets `cross_validation_path` fit each fold once.
    """

    def __init__(self,
                 estimator=None,
                 gamma=(1.,)):
        self.estimator = estimator
        self.gamma = gamma

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return np.stack([self.estimator_.predict(X, gamma=g) for g in self.gamma], axis=1)


@dataclass
class GLMNetControl(RegGLMControl):
    """
    Control parameters for GLMNet fitting.
    
    Parameters
    ----------
    fdev: float
        Fractional deviance tolerance for early stopping.
    mnlam: int
        Minimum number of lambda values fit before the path may stop early.
    devmax: float
        The path stops once the fraction of deviance explained exceeds this.
    logging: bool
        Write info and debug messages to log?

    As in R's `glmnet.path`, the early stopping rules are not used when
    `lambda_values` are given.
    """
    fdev: float = 1e-5
    mnlam: int = 5
    devmax: float = 0.999
    logging: bool = False


@dataclass
class GLMNetSpec(object):
    """
    Specification for GLMNet models.
    
    Parameters
    ----------
    lambda_values: Optional[np.ndarray]
        An array of `lambda` hyperparameters.
    lambda_min_ratio: float
        Ratio of lambda_max to smallest lambda.
        Used to set sequence of lamdba values.
        Values are equally spaced on a log-scale from lambda_max to
        lambda_max * lambda_min_ratio.
    nlambda: int
        Number of values on data-dependent grid of lambda values.
        Values are equally spaced on a log-scale from lambda_max to
        lambda_max * lambda_min_ratio.
    alpha: float
        The elasticnet mixing parameter in [0,1]. The penalty is
        defined as $(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.$
        `alpha=1` is the lasso penalty, and `alpha=0` the ridge
        penalty. Defaults to 1.
    lower_limits: float
        Vector of lower limits for each coefficient; default
        `-np.inf`. Each of these must be non-positive. Can be
        presented as a single value (which will then be replicated),
        else a vector of length `nvars`.
    upper_limits: float
        Vector of upper limits for each coefficient; default
        `np.inf`. See `lower_limits`.
    penalty_factor: Optional[Union[float, np.ndarray]]
        Separate penalty factors can be applied to each
        coefficient. This is a number that multiplies `lambda_val` to
        allow differential shrinkage. Can be 0 for some variables,
        which implies no shrinkage, and that variable is always
        included in the model. Default is 1 for all variables (and
        implicitly infinity for variables listed in `exclude`). Note:
        the penalty factors are internally rescaled to sum to
        `nvars=X.shape[1]`.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is True.
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    control: GLMNetControl
        Parameters to control the solver.
    regularized_estimator: BaseEstimator
        Estimator class used for fitting each point on path.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    relax: bool
        If True, also fit the relaxed lasso (R's `relax=TRUE`): each
        distinct active set along the path is refit without a penalty
        (lambda 0, other variables excluded). Predictions can then
        blend the two fits with `gamma`. Default is False.
    relax_maxp: Optional[int]
        Active sets with more than `relax_maxp` variables are not
        refit (R's `maxp` in `relax.glmnet`); defaults to `nobs - 3`.
    """
    lambda_values: Optional[np.ndarray] = None
    lambda_min_ratio: float = None
    nlambda: int = 100
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None
    fit_intercept: bool = True
    standardize: bool = True
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    control: GLMNetControl = field(default_factory=GLMNetControl)
    regularized_estimator: BaseEstimator = RegGLM
    offset_id: Union[str,int] = None
    weight_id: Union[str,int] = None
    response_id: Union[str,int] = None
    exclude: list = field(default_factory=list)
    relax: bool = False
    relax_maxp: Optional[int] = None


@dataclass
class GLMNet(BaseEstimator,
             GLMNetSpec):
    """
    GLMNet: Generalized Linear Models with Elastic Net regularization.
    
    Parameters
    ----------
    lambda_values: Optional[np.ndarray]
        An array of `lambda` hyperparameters.
    lambda_min_ratio: float
        Ratio of lambda_max to smallest lambda.
        Used to set sequence of lamdba values.
        Values are equally spaced on a log-scale from lambda_max to
        lambda_max * lambda_min_ratio.
    nlambda: int
        Number of values on data-dependent grid of lambda values.
        Values are equally spaced on a log-scale from lambda_max to
        lambda_max * lambda_min_ratio.
    alpha: float
        The elasticnet mixing parameter in [0,1]. The penalty is
        defined as $(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.$
        `alpha=1` is the lasso penalty, and `alpha=0` the ridge
        penalty. Defaults to 1.
    lower_limits: float
        Vector of lower limits for each coefficient; default
        `-np.inf`. Each of these must be non-positive. Can be
        presented as a single value (which will then be replicated),
        else a vector of length `nvars`.
    upper_limits: float
        Vector of upper limits for each coefficient; default
        `np.inf`. See `lower_limits`.
    penalty_factor: Optional[Union[float, np.ndarray]]
        Separate penalty factors can be applied to each
        coefficient. This is a number that multiplies `lambda_val` to
        allow differential shrinkage. Can be 0 for some variables,
        which implies no shrinkage, and that variable is always
        included in the model. Default is 1 for all variables (and
        implicitly infinity for variables listed in `exclude`). Note:
        the penalty factors are internally rescaled to sum to
        `nvars=X.shape[1]`.
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    standardize: bool
        Standardize columns of X according to weights? Default is True.
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    control: GLMNetControl
        Parameters to control the solver.
    regularized_estimator: BaseEstimator
        Estimator class used for fitting each point on path.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    relax: bool
        If True, also fit the relaxed lasso (R's `relax=TRUE`): each
        distinct active set along the path is refit without a penalty
        (lambda 0, other variables excluded). Predictions can then
        blend the two fits with `gamma`. Default is False.
    relax_maxp: Optional[int]
        Active sets with more than `relax_maxp` variables are not
        refit (R's `maxp` in `relax.glmnet`); defaults to `nobs - 3`.
    """

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
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
            regularizer=None,             # last 3 options non sklearn API
            warm_state=None,
            interpolation_grid=None):
        """
        Fit GLMNet model.

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
        regularizer: ElNetRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer.
        warm_state: GLMState, optional
            Warm start state.
        interpolation_grid: np.ndarray, optional
            Grid for interpolation of coefficients.

        Returns
        -------
        self: object
            GLMNet class instance.
        """
        if not hasattr(self, "_family"):
            self._family = self._finalize_family(response=y)

        X_fit, y_fit = X, y # as passed, for the relaxed refits

        self._set_penalty_factor(X, y)
        X, y, response, offset, weight = self.get_data_arrays(X, y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        n_samples, n_features = X.shape

        # we use column of y to retrieve optional weight

        sample_weight = weight
        self.normed_sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        self.reg_glm_est_ = self.regularized_estimator(
                               lambda_val=self.control.big,
                               family=self.family,
                               alpha=self.alpha,
                               penalty_factor=self.penalty_factor_,
                               lower_limits=self.lower_limits,
                               upper_limits=self.upper_limits,
                               fit_intercept=self.fit_intercept,
                               standardize=self.standardize,
                               control=self.control,
                               offset_id=self.offset_id,
                               weight_id=self.weight_id,            
                               response_id=self.response_id,
                               exclude=self.excluded_
                               )

        self.reg_glm_est_.fit(X,
                              y,
                              None,
                              fit_null=False,
                              warm_state=warm_state) 
        regularizer_ = self.reg_glm_est_.regularizer_

        state, keep_ = self._get_initial_state(X,
                                               y,
                                               self.excluded_)

        # _get_initial_state fits the unpenalized variables on the original
        # scale: put the state in the coordinates of the (standardized) design
        design = self.reg_glm_est_.design_
        state = state.__class__(state.coef * design.scaling_,
                                state.intercept + (state.coef * design.centers_).sum())
        state.update(design,
                     self._family,
                     offset)
        self.design_ = design
        
        logl_score = state.logl_score(self._family,
                                      response,
                                      normed_sample_weight)

        score_ = (design.T @ logl_score)[1:]
        pf = regularizer_.penalty_factor_
        score_ /= (pf + (pf <= 0))
        # excluded variables, including those with an infinite penalty factor
        score_[regularizer_.exclude] = 0
        self.lambda_max_ = np.fabs(score_).max() / max(self.alpha, 1e-3)

        if self.lambda_values is None:
            if self.lambda_min_ratio is None:
                lambda_min_ratio = 1e-2 if n_samples < n_features else 1e-4
            else:
                lambda_min_ratio = self.lambda_min_ratio
            self.lambda_values_ = np.exp(np.linspace(
                                          np.log(1),
                                          np.log(lambda_min_ratio),
                                          self.nlambda)
                                         )
            self.lambda_values_ *= self.lambda_max_
        else:
            self.lambda_values = np.asarray(self.lambda_values)
            self.lambda_values_ = np.sort(self.lambda_values)[::-1]
            self.nlambda = self.lambda_values.shape[0]
            self.lambda_min_ratio = (self.lambda_values.min() /
                                     self.lambda_values.max())

        coefs_ = []
        intercepts_ = []
        dev_ratios_ = []
        sample_weight_sum = sample_weight.sum()
        
        (null_fit,
         self.null_deviance_) = self._family.get_null_deviance(
                                    response=response,
                                    sample_weight=sample_weight,
                                    offset=offset,
                                    fit_intercept=self.fit_intercept)

        for l in self.lambda_values_:

            if self.control.logging: logging.info(f'Fitting parameter {l}')
            self.reg_glm_est_.lambda_val = regularizer_.lambda_val = l
            self.reg_glm_est_.fit(X,
                                  y,
                                  None, # normed_sample_weight,
                                  regularizer=regularizer_,
                                  check=False,
                                  fit_null=False)

            self.state_ = self.reg_glm_est_.state_

            coefs_.append(self.reg_glm_est_.coef_.copy())
            intercepts_.append(self.reg_glm_est_.intercept_)
            dev_ratios_.append(1 - self.reg_glm_est_.deviance_ / self.null_deviance_)
            if self._stop_path(dev_ratios_):
                break
            
        self.coefs_ = np.array(coefs_)
        self.intercepts_ = np.array(intercepts_)

        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = (self.coefs_ != 0).sum(1)
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        nfit = self.coefs_.shape[0]

        self.lambda_values_ = self.lambda_values_[:nfit]

        if self.relax:
            self._fit_relaxed(X_fit, y_fit)

        if interpolation_grid is not None:
            self._interpolate_fit(interpolation_grid)

        self.coef_path_ = CoefPath(
            coefs=self.coefs_,
            intercepts=self.intercepts_,
            lambda_values=self.lambda_values_,
            feature_names=self.feature_names_in_,
            fracdev=np.array(dev_ratios_)
        )

        return self
    
    def predict(self,
                X,
                prediction_type='response',
                interpolation_grid=None,
                offset=None,
                gamma=1.):
        """
        Predict using the fitted GLMNet model.

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
        interpolation_grid: np.ndarray, optional
            Grid of lambda values for interpolation. If provided, coefficients are interpolated
            to these values before prediction.
        offset: np.ndarray, optional
            Offset for the rows of `X`, of shape `(nobs,)`, added to the linear
            predictor (R's `newoffset`). If the model was fit with `offset_id`,
            pass the offset for the new data here; if omitted, no offset is
            used.
        gamma: float, optional
            Blend of the lasso (1, the default) and relaxed (0) fits, as R's
            `predict(..., gamma=)`; requires `relax=True` unless 1.

        Returns
        -------
        np.ndarray
            Predictions for each lambda value.
        """

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
        if prediction_type != 'link':
            fits = self._family.predict(linear_pred_, prediction_type=prediction_type)
        else:
            fits = linear_pred_

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
            nlambda = self.nlambda
            squeeze = False
            
        value = np.empty((fits.shape[0], nlambda), fits.dtype)
        value[:,:fits.shape[1]] = fits
        value[:,fits.shape[1]:] = fits[:,-1][:,None]
        if squeeze:
            value = np.squeeze(value)
        return value
        
    def interpolate_coefs(self,
                          interpolation_grid,
                          gamma=1.):
        """
        Interpolate coefficients to a new lambda grid.

        Parameters
        ----------
        interpolation_grid: np.ndarray
            New lambda values for interpolation.
        gamma: float
            Blend of the lasso (`gamma=1`, the default) and relaxed
            (`gamma=0`) fits, as R's `coef(..., gamma=)`; requires
            `relax=True` unless 1.

        Returns
        -------
        tuple
            (coefs_, intercepts_) interpolated to the new grid.
        """
        coefs_, intercepts_ = self._blended_coefs(gamma)
        return self._interpolate(coefs_, intercepts_, interpolation_grid)

    def _interpolate(self,
                     coefs,
                     intercepts,
                     interpolation_grid):
        """
        Interpolate `coefs` and `intercepts`, whose rows correspond to
        `lambda_values_`, to `interpolation_grid` (linearly in the index
        of the lambda values).
        """
        L = self.lambda_values_
        interpolation_grid = np.asarray(interpolation_grid)
        shape = interpolation_grid.shape
        interpolation_grid = np.atleast_1d(interpolation_grid)
        interpolation_grid = np.clip(interpolation_grid, L.min(), L.max())
        idx_ = interp1d(L, np.arange(L.shape[0]).astype(float))(interpolation_grid)
        coefs_ = []
        intercepts_ = []

        for v_ in idx_:
            v_ceil = int(np.ceil(v_))
            w_ = (v_ceil - v_)
            if v_ceil > 0:
                coefs_.append(coefs[v_ceil] * (1 - w_) + w_ * coefs[v_ceil-1])
                intercepts_.append(intercepts[v_ceil] * (1 - w_) + w_ * intercepts[v_ceil-1])
            else:
                coefs_.append(coefs[0])
                intercepts_.append(intercepts[0])

        if shape == interpolation_grid.shape:
            return np.asarray(coefs_), np.asarray(intercepts_)
        else:
            return np.asarray(coefs_)[0], np.asarray(intercepts_)[0]

    def _interpolate_fit(self,
                         interpolation_grid):
        """
        Replace the fitted path (and the relaxed fits, if any) by its
        interpolation to `interpolation_grid`.
        """
        if self.relax:
            relaxed = self._interpolate(self.relaxed_coefs_,
                                        self.relaxed_intercepts_,
                                        interpolation_grid)
        self.coefs_, self.intercepts_ = self.interpolate_coefs(interpolation_grid)
        if self.relax:
            self.relaxed_coefs_, self.relaxed_intercepts_ = relaxed

    def _blended_coefs(self,
                       gamma):
        """
        Coefficients and intercepts along the path for `gamma`:
        `gamma * lasso + (1 - gamma) * relaxed`, as R's `blend.relaxed`.
        """
        gamma = _check_gamma(gamma)
        if gamma == 1:
            return self.coefs_, self.intercepts_
        if not hasattr(self, 'relaxed_coefs_'):
            raise ValueError('gamma < 1 requires a relaxed fit: fit with relax=True')
        # as R's blend.relaxed
        gamma = max(gamma, 1e-5)
        return (gamma * self.coefs_ + (1 - gamma) * self.relaxed_coefs_,
                gamma * np.asarray(self.intercepts_) + (1 - gamma) * self.relaxed_intercepts_)

    def _fit_relaxed(self,
                     X,
                     y):
        """
        The relaxed fits, as R's `relax.glmnet`: each distinct active set
        along the path is refit at lambda 0 with the other variables
        excluded, by a clone of this estimator (so weights, offsets, limits
        and control carry over). Sets with more than `relax_maxp` variables
        are not refit; lambdas with such sets use the relaxed fit of the
        last lambda that was refit. Where a refit finds no solution (e.g.
        separable classes) the lasso solution is kept, with a warning.

        Sets `relaxed_coefs_`, `relaxed_intercepts_` (shaped as `coefs_` and
        `intercepts_`), `relaxed_fracdev_` and `relaxed_omitted_` (True for
        lambdas whose active set was not refit).
        """
        coefs = np.asarray(self.coefs_)
        intercepts = np.asarray(self.intercepts_)
        # active set at each lambda; for multi-response, the union over responses
        active = coefs != 0
        if active.ndim == 3:
            active = active.any(-1)
        nobs = X.shape[0]
        maxp = nobs - 3 if self.relax_maxp is None else self.relax_maxp

        relaxed_coefs = coefs.copy()
        relaxed_intercepts = intercepts.astype(float)
        fracdev = np.asarray(self.summary_['Fraction Deviance Explained'], float).copy()
        omitted = active.sum(1) > maxp

        refits = {}
        failed = []
        for k in np.nonzero(~omitted & active.any(1))[0]:
            key = active[k].tobytes()
            if key not in refits:
                refit = clone(self)
                refit.relax = False
                refit.lambda_values = np.array([0.])
                refit.exclude = sorted(set(self.exclude) |
                                       set(np.nonzero(~active[k])[0].tolist()))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    refit.fit(X, y)
                if np.asarray(refit.coefs_).shape[0] == 0:
                    # no solution at lambda 0, e.g. separable classes
                    refits[key] = None
                else:
                    for w in caught:
                        warnings.warn(w.message, w.category)
                    refits[key] = (refit.coefs_[-1],
                                   refit.intercepts_[-1],
                                   refit.summary_['Fraction Deviance Explained'].iloc[-1])
            if refits[key] is None:
                failed.append(k)
            else:
                relaxed_coefs[k], relaxed_intercepts[k], fracdev[k] = refits[key]

        if failed:
            warnings.warn(f'the unpenalized refit did not converge at {len(failed)} lambda values '
                          f'(indices {failed}); the relaxed fit uses the lasso solution there')

        if omitted.any():
            kept = np.nonzero(~omitted)[0]
            if kept.size == 0:
                raise ValueError(f'every active set has more than relax_maxp={maxp} variables')
            last = kept.max()
            relaxed_coefs[omitted] = relaxed_coefs[last]
            relaxed_intercepts[omitted] = relaxed_intercepts[last]
            fracdev[omitted] = fracdev[last]

        self.relaxed_coefs_ = relaxed_coefs
        self.relaxed_intercepts_ = relaxed_intercepts
        self.relaxed_fracdev_ = fracdev
        self.relaxed_omitted_ = omitted

    def nonzero(self,
                interpolation_grid=None):
        """
        Indices of the nonzero coefficients along the path, as
        `predict(fit, type="nonzero")` in R.

        Parameters
        ----------
        interpolation_grid: np.ndarray, optional
            Grid of lambda values. If provided, coefficients are interpolated
            to these values first, as in `predict`.

        Returns
        -------
        list or np.ndarray
            For each lambda in `lambda_values_` (or in `interpolation_grid`),
            the (0-based) indices of the features with a nonzero coefficient.
            A single array if `interpolation_grid` is a scalar.
        """
        coefs_, squeeze = self._nonzero_coefs(interpolation_grid)
        value = [np.nonzero(c)[0] for c in coefs_]
        return value[0] if squeeze else value

    def _nonzero_coefs(self,
                       interpolation_grid):
        """
        Coefficients along the path, or interpolated to `interpolation_grid`,
        with a leading lambda axis; and whether the grid was a scalar.
        """
        check_is_fitted(self, ["coefs_"])
        if interpolation_grid is None:
            return self.coefs_, False
        grid_ = np.asarray(interpolation_grid)
        coefs_, _ = self.interpolate_coefs(np.atleast_1d(grid_))
        return coefs_, grid_.ndim == 0

    def refit_path(self,
                   X,
                   y,
                   lambda_val):
        """
        Refit the path with `lambda_val` added to its lambda values.

        As R's ``update(object, lambda=...)`` in ``predict.glmnet(...,
        exact=TRUE)``: the new path is fit to the values of `lambda_values_`
        and `lambda_val` combined, so its solutions at `lambda_val` are exact
        rather than interpolated.

        Parameters
        ----------
        X : Union[np.ndarray, scipy.sparse, DesignSpec]
            Feature matrix used in `fit`.
        y : np.ndarray or pd.DataFrame
            Response used in `fit`, with any weight or offset columns.
        lambda_val : float or np.ndarray
            Value(s) of lambda to add to the path.

        Returns
        -------
        GLMNet
            A new fitted estimator; `self` is unchanged.
        """
        check_is_fitted(self, ["coefs_"])

        lambda_val = np.atleast_1d(np.asarray(lambda_val, float))
        if np.any(lambda_val < 0):
            raise ValueError('lambda values must be non-negative')
        refit = clone(self)
        if np.all(np.isin(lambda_val, self.lambda_values_)):
            refit.lambda_values = self.lambda_values_.copy()
        else:
            refit.lambda_values = np.unique(np.concatenate([lambda_val, self.lambda_values_]))[::-1]
        return refit.fit(X, y)

    def exact_coefs(self,
                    X,
                    y,
                    lambda_val,
                    gamma=1.):
        """
        Coefficients at `lambda_val` from refitting the path, rather than
        interpolating; R's ``coef(..., s=lambda_val, exact=TRUE)``.

        Parameters
        ----------
        X : Union[np.ndarray, scipy.sparse, DesignSpec]
            Feature matrix used in `fit`.
        y : np.ndarray or pd.DataFrame
            Response used in `fit`, with any weight or offset columns.
        lambda_val : float or np.ndarray
            Value(s) of lambda.
        gamma : float
            Blend of the lasso (1, the default) and relaxed (0) fits, as
            in `interpolate_coefs`; requires `relax=True` unless 1.

        Returns
        -------
        tuple
            (coefs, intercepts) at `lambda_val`, shaped as by `interpolate_coefs`.
        """
        return self.refit_path(X, y, lambda_val).interpolate_coefs(lambda_val, gamma=gamma)

    def relaxed_coef_path(self,
                          gamma=0.):
        """
        The coefficient path of the fit blended with `gamma` (the relaxed
        fit for `gamma=0`), for plotting as R's ``plot(fit, gamma=)``.

        Parameters
        ----------
        gamma : float
            Blend of the lasso (1) and relaxed (0, the default) fits.

        Returns
        -------
        CoefPath
        """
        check_is_fitted(self, ["coefs_"])
        coefs_, intercepts_ = self._blended_coefs(gamma)
        fracdev = np.asarray(self.summary_['Fraction Deviance Explained'])
        if gamma < 1:
            # as R's blend.relaxed
            g = max(gamma, 1e-5)
            fracdev = g * fracdev + (1 - g) * self.relaxed_fracdev_[:fracdev.shape[0]]
        return CoefPath(coefs=coefs_,
                        intercepts=intercepts_,
                        lambda_values=self.lambda_values_,
                        feature_names=self.feature_names_in_,
                        fracdev=fracdev)

    def cv_choice(self,
                  which='1se',
                  score=None):
        """
        The lambda (and gamma) chosen by the last `cross_validation_path`,
        as R's ``lambda.1se`` / ``lambda.min`` of a `cv.glmnet` fit (and
        ``gamma.1se`` / ``gamma.min`` for a relaxed fit).

        Parameters
        ----------
        which : str
            '1se' (the default, as R's ``s="lambda.1se"``) or 'best'
            (R's ``"lambda.min"``).
        score : str, optional
            Name of the score the choice is based on; defaults to the
            family's first default score (as in `ScorePath.plot`).

        Returns
        -------
        tuple
            (lambda, gamma); gamma is 1 unless the cross-validation was of a
            relaxed fit.
        """
        if which not in ['1se', 'best']:
            raise ValueError("which should be one of '1se' or 'best'")
        if not hasattr(self, 'score_path_'):
            raise ValueError('run cross_validation_path first')
        if score is None:
            score = self._family._default_scorers()[0].name
        relaxed = getattr(self, 'relaxed_score_path_', None)
        if self.relax and relaxed is not None:
            index = relaxed.index_1se if which == '1se' else relaxed.index_best
            if score not in index.index:
                raise ValueError(f'no cross-validated score named {score!r}')
            return float(index.loc[score, 'lambda']), float(index.loc[score, 'gamma'])
        index = self.score_path_.index_1se if which == '1se' else self.score_path_.index_best
        if index is None or score not in index.index:
            raise ValueError(f'no cross-validated score named {score!r}')
        return float(index[score]), 1.

    def cv_coefs(self,
                 which='1se',
                 score=None):
        """
        Coefficients at the lambda (and gamma) chosen by the last
        `cross_validation_path`, as R's ``coef(cvfit, s="lambda.1se")``.

        Parameters
        ----------
        which : str
            '1se' (the default) or 'best'; see `cv_choice`.
        score : str, optional
            Score the choice is based on; see `cv_choice`.

        Returns
        -------
        tuple
            (coefs, intercepts), as from `interpolate_coefs` at a single lambda.
        """
        lambda_val, gamma = self.cv_choice(which=which, score=score)
        return self.interpolate_coefs(lambda_val, gamma=gamma)

    def cv_predict(self,
                   X,
                   which='1se',
                   score=None,
                   **predict_args):
        """
        Predictions at the lambda (and gamma) chosen by the last
        `cross_validation_path`, as R's ``predict(cvfit, newx,
        s="lambda.1se")``.

        Parameters
        ----------
        X : Union[np.ndarray, scipy.sparse, DesignSpec]
            Feature matrix to predict.
        which : str
            '1se' (the default) or 'best'; see `cv_choice`.
        score : str, optional
            Score the choice is based on; see `cv_choice`.
        predict_args :
            Other arguments of `predict`, e.g. `prediction_type` or `offset`.

        Returns
        -------
        np.ndarray
            Predictions, as from `predict` with a scalar `interpolation_grid`.
        """
        lambda_val, gamma = self.cv_choice(which=which, score=score)
        return self.predict(X,
                            interpolation_grid=lambda_val,
                            gamma=gamma,
                            **predict_args)

    def cross_validation_path(self,
                              X,
                              y,
                              cv=10,
                              groups=None,
                              n_jobs=None,
                              verbose=0,
                              fit_params={},
                              pre_dispatch='2*n_jobs',
                              alignment='lambda',
                              scorers=[], # GLMScorer instances
                              gamma=None):
        """
        Perform cross-validation along the regularization path.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(nobs, nvars)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        cv: int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy.
        groups: array-like, optional
            Group labels for the samples used while splitting the dataset into train/test set.
        n_jobs: int, optional
            Number of jobs to run in parallel.
        verbose: int
            The verbosity level.
        fit_params: dict, optional
            Parameters to pass to the fit method of the estimator.
        pre_dispatch: str, optional
            Controls the number of jobs that get dispatched during parallel execution.
        alignment: str
            One of 'lambda' or 'fraction'. How to align predictions across folds.
        scorers: list
            List of GLMScorer instances.
        gamma: sequence of float, optional
            For a relaxed fit (`relax=True`), the values of `gamma` to
            cross-validate along with lambda, as R's `cv.glmnet(relax=TRUE,
            gamma=)`. Defaults to `(0, 0.25, 0.5, 0.75, 1)`. Only used with
            `relax=True`.

        Returns
        -------
        tuple
            (predictions, score_path_)
            predictions: np.ndarray
                Cross-validated predictions for each sample and lambda value.
            score_path_: ScorePath
                An object containing cross-validation results, including scores (as a DataFrame),
                standard errors, best/1se indices, lambda values, and more. Access scores via
                score_path_.scores, e.g. score_path_.scores['Mean Squared Error'].

            For a relaxed fit, predictions has an axis for `gamma` after the
            first, and a `RelaxedScorePath` is returned, with a `ScorePath`
            for each value of `gamma` and the best (lambda, gamma) pairs. It
            is also stored as `relaxed_score_path_`; `score_path_` holds the
            results for the lasso (`gamma=1`), as without `relax`.
        """
        check_is_fitted(self, ["coefs_"])

        if alignment not in ['lambda', 'fraction']:
            raise ValueError("alignment must be one of 'lambda' or 'fraction'")
        if gamma is not None and not self.relax:
            raise ValueError('gamma requires a relaxed fit: fit with relax=True')

        cloned_path = clone(self)
        fit_params = dict(fit_params)
        if alignment == 'lambda':
            fit_params.update(interpolation_grid=self.lambda_values_)
        else:
            if self.lambda_values is not None:
                warnings.warn('Using pre-specified lambda values, not proportional to lambda_max')
                cloned_path.lambda_values = cloned_path.lambda_values[:self.lambda_values_.shape[0]]
            fit_params = {}

        if self.relax:
            if gamma is None:
                gamma = (0, 0.25, 0.5, 0.75, 1)
            gamma = sorted(set(_check_gamma(g) for g in np.atleast_1d(gamma)))
            # as R's cv.relaxed, also compute the lasso's (gamma=1) scores
            all_gamma = gamma + ([] if 1 in gamma else [1.])
            estimator = _RelaxedPredictor(estimator=cloned_path,
                                          gamma=tuple(all_gamma))
        else:
            estimator = cloned_path

        X, y, groups = indexable(X, y, groups)
        
        cv = check_cv(cv, y, classifier=False)

        predictions = cross_val_predict(estimator,
                                        X,
                                        y,
                                        groups=groups,
                                        cv=cv,
                                        n_jobs=n_jobs,
                                        verbose=verbose,
                                        params=fit_params,
                                        pre_dispatch=pre_dispatch)

        response, offset, weight = self.get_data_arrays(X, y, check=False)[2:]
        splits = [test for _, test in cv.split(np.arange(X.shape[0]))]
        nlambda = self.lambda_values_.shape[0]

        def score(predictions, gamma):
            # truncate to the size we got
            return self._cv_score_path(predictions[:,:nlambda],
                                       response,
                                       offset,
                                       weight,
                                       y,
                                       splits,
                                       scorers,
                                       gamma)

        if not self.relax:
            predictions, self.score_path_ = score(predictions, 1.)
            return predictions, self.score_path_

        results = [score(predictions[:,i], g) for i, g in enumerate(all_gamma)]
        self.score_path_ = results[all_gamma.index(1.)][1]
        results = results[:len(gamma)]
        score_paths = [path for _, path in results]
        index_best_, index_1se_ = _tune_relaxed(score_paths,
                                                gamma,
                                                list(set(scorers).union(self._family._default_scorers())))
        self.relaxed_score_path_ = RelaxedScorePath(gamma=np.asarray(gamma),
                                                    score_paths=score_paths,
                                                    index_best=index_best_,
                                                    index_1se=index_1se_)
        predictions = np.stack([preds for preds, _ in results], axis=1)
        return predictions, self.relaxed_score_path_

    def _cv_score_path(self,
                       predictions,
                       response,
                       offset,
                       weight,
                       y,
                       splits,
                       scorers,
                       gamma):
        """
        Score cross-validated `predictions` (of the path blended with
        `gamma`), adding the offset of the held-out rows.

        Returns
        -------
        tuple
            (predictions, ScorePath), the predictions adjusted for the offset.
        """
        # adjust for offset
        # because predictions are just X\beta

        if offset is not None:
            predictions = self._offset_predictions(predictions,
                                                   offset)

        scorer = PathScorer(predictions=predictions,
                            sample_weight=weight,
                            data=(response, y),
                            splits=splits,
                            family=self._family,
                            index=self.lambda_values_,
                            complexity_order='increasing',
                            compute_std_error=True)

        (cv_scores_,
         index_best_,
         index_1se_) = scorer.compute_scores(scorers=scorers)

        coefs_, _ = self._blended_coefs(gamma)
        fracdev = np.asarray(self.summary_['Fraction Deviance Explained'])
        if gamma < 1:
            # as R's blend.relaxed
            g = max(gamma, 1e-5)
            fracdev = g * fracdev + (1 - g) * self.relaxed_fracdev_[:fracdev.shape[0]]

        return predictions, ScorePath(scores=cv_scores_,
                                      index_best=index_best_,
                                      index_1se=index_1se_,
                                      lambda_values=self.lambda_values_,
                                      norm=np.fabs(coefs_).sum(1),
                                      fracdev=fracdev,
                                      family=self._family)
    
    def score_path(self,
                   X,
                   y,
                   scorers=[],
                   plot=True):
        """
        Compute scores for a fitted regularization path on provided data.

        This method evaluates the fitted GLMNet model's path (for all lambda values)
        on the given data using one or more scoring metrics. It does not perform cross-validation;
        instead, it computes scores for the entire dataset (or a provided split) as a single group.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix to score, shape (n_samples, n_features).
        y : array-like
            Target values or structured data for scoring.
        scorers : list, optional
            List of GLMScorer instances or compatible scoring objects. If empty, uses default scorers for the family.
        plot : bool, optional
            If True, may trigger plotting of the score path (not implemented in this method, but available via ValidationPath.plot).

        Returns
        -------
        ValidationPath
            An object containing the computed scores for each lambda value, as well as indices for best/1se selection, lambda values, norms, and deviance explained. Use the .scores attribute to access the DataFrame of scores.

        Examples
        --------
        >>> model.fit(X, y)
        >>> val_path = model.score_path(X, y)
        >>> val_path.scores['Mean Squared Error']
        """
        check_is_fitted(self, ["coefs_"])

        predictions = self.predict(X, interpolation_grid=self.lambda_values_)
        response, offset, weight = clone(self).get_data_arrays(X, y, check=False)[2:]

        # as in cross_validation_path: predictions are just X\beta
        if offset is not None:
            predictions = self._offset_predictions(predictions,
                                                   offset)

        splits = [np.arange(X.shape[0])]

        scorer = PathScorer(predictions=predictions,
                            sample_weight=weight,
                            data=(response, y),
                            splits=splits,
                            family=self._family,
                            index=self.lambda_values_,
                            complexity_order='increasing',
                            compute_std_error=False)

        (scores_,
         index_best_,
         index_1se_) = scorer.compute_scores(scorers=scorers)

        return ScorePath(scores=scores_,
                          index_best=index_best_,
                          index_1se=index_1se_,
                          lambda_values=self.lambda_values_,
                          norm=np.fabs(self.coefs_).sum(1),
                          fracdev=self.summary_['Fraction Deviance Explained'],
                          family=self._family)

    def _offset_predictions(self,
                            predictions,
                            offset):
        """
        Adjust predictions for offset.

        Parameters
        ----------
        predictions: np.ndarray
            Raw predictions.
        offset: np.ndarray
            Offset values.

        Returns
        -------
        np.ndarray
            Adjusted predictions.
        """
        linpred = self._family.link(predictions) + offset[:, None]
        return self._family.predict(linpred, prediction_type='response')
   
    def _stop_path(self,
                   dev_ratios):
        """
        Whether to stop the path after the fits with fractions of deviance
        explained `dev_ratios`, as R's `glmnet.path`: never before
        `control.mnlam` fits or when `lambda_values` were given; otherwise
        once the fraction exceeds `control.devmax`, or it stops increasing
        (relative to `control.fdev`, with R's rules for the gaussian and
        poisson families).
        """
        control = self.control
        k = len(dev_ratios)
        mnl = min(self.nlambda, getattr(control, 'mnlam', 5))
        if self.lambda_values is not None or k < mnl:
            return False
        if dev_ratios[-1] > getattr(control, 'devmax', 0.999):
            return True
        if k == 1:
            return False
        base = getattr(self._family, 'base', None) # e.g. Cox has none
        if isinstance(base, sm_family.Gaussian):
            return dev_ratios[-1] - dev_ratios[-2] < control.fdev * dev_ratios[-1]
        if isinstance(base, sm_family.Poisson):
            return dev_ratios[-1] - dev_ratios[k - mnl] < 10 * control.fdev * dev_ratios[-1]
        return dev_ratios[-1] - dev_ratios[-2] < control.fdev

    def _get_initial_state(self,
                           X,
                           y,
                           exclude):
        """
        Get initial state for fitting.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix.
        y: np.ndarray
            Response variable.
        exclude: list
            Indices of variables to exclude.

        Returns
        -------
        tuple
            (state, keep) where state is GLMState and keep is boolean array.
        """
        n_samples, n_features = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor_ == 0
        keep[exclude] = 0

        coef_ = np.zeros(n_features)

        if keep.sum() > 0:
            X_keep = X[:,keep]

            glm = GLM(fit_intercept=self.fit_intercept,
                      family=self.family,
                      offset_id=self.offset_id,
                      weight_id=self.weight_id,
                      response_id=self.response_id,
                      control=self.control)
            glm.fit(X_keep, y)
            coef_[keep] = glm.coef_
            intercept_ = glm.intercept_
        else:
            if self.fit_intercept:
                response, offset, weight = self.get_data_arrays(X, y, check=False)[2:]
                state0 = self._family.null_fit(response,
                                               fit_intercept=self.fit_intercept,
                                               sample_weight=weight,
                                               offset=offset)
                intercept_ = state0.coef[0] # null state has no intercept
                                            # X a column of 1s
            else:
                intercept_ = 0
        return GLMState(coef=coef_, intercept=intercept_), keep.astype(float)

    def get_GLM(self,
                ridge_coef=0):
        """
        Get a GLM instance with the same parameters.

        Parameters
        ----------
        ridge_coef: float
            Ridge coefficient.

        Returns
        -------
        GLM
            GLM instance.
        """
        return GLM(family=self.family,
                   fit_intercept=self.fit_intercept,
                   standardize=self.standardize,
                   ridge_coef=ridge_coef,
                   offset_id=self.offset_id,
                   weight_id=self.weight_id,
                   response_id=self.response_id)

    def get_fixed_lambda(self,
                         lambda_val):
        """
        Get a regularized estimator for a fixed lambda value.

        Parameters
        ----------
        lambda_val: float
            Lambda value.

        Returns
        -------
        tuple
            (estimator, state) where estimator is the regularized estimator
            and state is the fitted state.
        """
        check_is_fitted(self, ["coefs_", "feature_names_in_"])

        estimator = self.regularized_estimator(
                               lambda_val=lambda_val,
                               family=self._fixed_lambda_family(),
                               alpha=self.alpha,
                               penalty_factor=self.penalty_factor_,
                               lower_limits=self.lower_limits,
                               upper_limits=self.upper_limits,
                               fit_intercept=self.fit_intercept,
                               standardize=self.standardize,
                               control=self.control,
                               offset_id=self.offset_id,
                               weight_id=self.weight_id,            
                               response_id=self.response_id,
                               exclude=self.excluded_
                               )

        coefs, intercepts = self.interpolate_coefs([lambda_val])
        cls = self.state_.__class__
        state = cls(coefs[0], intercepts[0])
        return estimator, state

    def _fixed_lambda_family(self):
        """Family passed to `regularized_estimator` by `get_fixed_lambda`."""
        return self.family

    def prefilter(self, X, y):
        """
        Deprecated: override `get_penalty_factor` instead, giving the
        variables to exclude an infinite penalty factor.

        The indices a `prefilter` override returns are still excluded (the
        default `get_penalty_factor` gives them an infinite factor), with a
        `FutureWarning`.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.

        Returns
        -------
        filtered : list
            List of feature indices to exclude.
        """
        return []

    def get_penalty_factor(self, X, y):
        """
        Penalty factors for a fit, computed from its data. Override in a
        subclass for penalty factors or exclusions that depend on the data,
        as R's glmnet allows functions for `penalty.factor` and `exclude`.
        It is called at the start of each `fit`, so it is re-run on each
        training fold in cross-validation, as in R's `cv.glmnet`.

        Parameters
        ----------
        X : array-like
            Feature matrix, as passed to `fit`.
        y : array-like
            Response, as passed to `fit` (possibly with weight and offset
            columns).

        Returns
        -------
        penalty_factor : Optional[Union[float, np.ndarray]]
            Penalty factors, as for `penalty_factor`: variables with an
            infinite factor are excluded. Defaults to `self.penalty_factor`
            (and an infinite factor for the indices returned by a deprecated
            `prefilter` override).
        """
        penalty_factor = self.penalty_factor
        if type(self).prefilter is not GLMNet.prefilter:
            warnings.warn('prefilter is deprecated: override get_penalty_factor instead, '
                          'giving the variables to exclude an infinite penalty factor',
                          FutureWarning)
            excluded = list(self.prefilter(X, y))
            if excluded:
                nvars = X.shape[1]
                penalty_factor = (np.ones(nvars) if penalty_factor is None else
                                  np.array(np.broadcast_to(penalty_factor, (nvars,)), dtype=float))
                penalty_factor[excluded] = np.inf
        return penalty_factor

    def _set_penalty_factor(self, X, y):
        """
        Set `penalty_factor_` and `excluded_` for a fit from
        `get_penalty_factor`: the variables in `exclude` or with an
        infinite factor are in `excluded_`, and `penalty_factor_` has the
        factors used by the solvers (an infinite factor replaced by 1, as in
        R's glmnet).
        """
        penalty_factor = self.get_penalty_factor(X, y)
        excluded = set(np.asarray(self.exclude, int).tolist())
        if penalty_factor is not None:
            penalty_factor = np.array(np.broadcast_to(np.asarray(penalty_factor, dtype=float),
                                                      (X.shape[1],)))
            infinite = np.isinf(penalty_factor)
            excluded |= set(np.nonzero(infinite)[0].tolist())
            penalty_factor[infinite] = 1
        self.excluded_ = sorted(excluded)
        self.penalty_factor_ = penalty_factor


@dataclass
class CoefPath(object):
    """
    Container for coefficient paths along the regularization path.

    Stores the coefficients, intercepts, lambda values, and feature names for each step in the path.
    Provides a plot method to visualize the coefficient trajectories as a function of lambda, norm, or deviance explained.

    Attributes
    ----------
    coefs : np.ndarray
        Array of coefficients for each lambda value (n_lambdas, n_features).
    intercepts : np.ndarray
        Array of intercepts for each lambda value (n_lambdas,).
    lambda_values : np.ndarray
        Array of lambda values along the path.
    feature_names : list or np.ndarray
        Names of the features (columns).
    fracdev : np.ndarray, optional
        Fraction of deviance explained at each lambda value.
    """
    coefs: np.ndarray
    intercepts: np.ndarray
    lambda_values: np.ndarray
    feature_names: list | np.ndarray
    fracdev: np.ndarray | None = None

    def plot(self,
             xvar='-lambda',
             ax=None,
             legend=False,
             drop=None,
             keep=None,
             label=False):
        """
        Plot coefficient paths.

        Parameters
        ----------
        xvar: str
            Variable to plot on x-axis. One of 'lambda', '-lambda', 'norm', 'dev'.
        ax: matplotlib.axes.Axes, optional
            Axes to plot on.
        legend: bool
            Whether to show legend.
        drop: list, optional
            Features to drop from the plot.
        keep: list, optional
            Features to keep in the plot.
        label: bool
            Label each curve with its feature name at the end of the
            path (the smallest lambda), as R's `plot(fit, label=TRUE)`.
            Features that are zero along the whole path are not labelled.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.
        """
        if xvar == '-lambda':
            index = pd.Index(-np.log(self.lambda_values))
            index.name = r'$-\log(\lambda)$'
        elif xvar == 'lambda':
            index = pd.Index(np.log(self.lambda_values))
            index.name = r'$\log(\lambda)$'
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            if self.fracdev is None:
                raise ValueError("fracdev must be set to use xvar='dev'")
            index = pd.Index(self.fracdev)
            index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be one of 'lambda', '-lambda', 'norm', 'dev'")

        coefs_ = self.coefs
        if coefs_.ndim > 2:
            # compute the l2 norm
            coefs_ = np.sqrt((coefs_**2).sum(-1))
            ylabel = r'Coefficient norms ($\|\beta\|_2$)'
        else:
            ylabel = r'Coefficients ($\beta$)'
        soln_path = pd.DataFrame(coefs_,
                                 columns=self.feature_names,
                                 index=index)
        if drop is not None:
            soln_path = soln_path.drop(columns=drop)
        if keep is not None:
            soln_path = soln_path.loc[:, keep]
        n_lines = 0 if ax is None else len(ax.get_lines())
        ax = soln_path.plot(ax=ax, legend=False)
        lines = ax.get_lines()[n_lines:n_lines + soln_path.shape[1]]
        ax.set_xlabel(index.name)
        ax.set_ylabel(ylabel)
        ax.axhline(0, c='k', ls='--')

        if label:
            # label at the end of the path, on the outside of the curves
            x_end = soln_path.index[-1]
            ha = 'left' if x_end >= soln_path.index[0] else 'right'
            for name, line in zip(soln_path.columns, lines):
                if np.all(soln_path[name] == 0):
                    continue
                ax.annotate(str(name),
                            (x_end, soln_path[name].iloc[-1]),
                            xytext=(3 if ha == 'left' else -3, 0),
                            textcoords='offset points',
                            ha=ha,
                            va='center',
                            fontsize='small',
                            color=line.get_color())

        if legend:
            fig = ax.figure
            if hasattr(fig, 'get_layout_engine') and fig.get_layout_engine() is not None:
                import warnings
                warnings.warn('If plotting a legend, layout of figure will be set to "constrained".')
            if hasattr(fig, 'set_layout_engine'):
                fig.set_layout_engine('constrained')
            fig.legend(loc='outside right upper')
        return ax
