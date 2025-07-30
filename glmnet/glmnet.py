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
from ._utils import _get_data
from .scorer import (PathScorer,
                     ScorePath)



@dataclass
class GLMNetControl(RegGLMControl):
    """
    Control parameters for GLMNet fitting.
    
    Parameters
    ----------
    fdev: float
        Fractional deviance tolerance for early stopping.
    logging: bool
        Write info and debug messages to log?
    """
    fdev: float = 1e-5
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

        self.excluded_ = copy(self.exclude)
        self.excluded_.extend(list(self.prefilter(X, y)))
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
                               penalty_factor=self.penalty_factor,
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

        state.update(self.reg_glm_est_.design_,
                     self._family,
                     offset)
        self.design_ = self.reg_glm_est_.design_
        
        logl_score = state.logl_score(self._family,
                                      response,
                                      normed_sample_weight)

        score_ = (self.reg_glm_est_.design_.T @ logl_score)[1:]
        pf = regularizer_.penalty_factor_
        score_ /= (pf + (pf <= 0))
        score_[self.excluded_] = 0
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
            if len(dev_ratios_) > 1:
                if self._family.is_gaussian:
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev * dev_ratios_[-1]:
                        break
                else: # TODO Poisson case
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev:
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
        
        if interpolation_grid is not None:
            self.coefs_, self.intercepts_ = self.interpolate_coefs(interpolation_grid)
           
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
                interpolation_grid=None):
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

        Returns
        -------
        np.ndarray
            Predictions for each lambda value.
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
                          interpolation_grid):
        """
        Interpolate coefficients to a new lambda grid.

        Parameters
        ----------
        interpolation_grid: np.ndarray
            New lambda values for interpolation.

        Returns
        -------
        tuple
            (coefs_, intercepts_) interpolated to the new grid.
        """
        L = self.lambda_values_
        interpolation_grid = np.asarray(interpolation_grid)
        shape = interpolation_grid.shape
        interpolation_grid = np.atleast_1d(interpolation_grid)
        interpolation_grid = np.clip(interpolation_grid, L.min(), L.max())
        idx_ = interp1d(L, np.arange(L.shape[0]).astype(float))(interpolation_grid)
        coefs_ = []
        intercepts_ = []

        ws = []
        for v_ in idx_:
            v_ceil = int(np.ceil(v_))
            w_ = (v_ceil - v_)
            ws.append(w_)
            if v_ceil > 0:
                coefs_.append(self.coefs_[v_ceil] * (1 - w_) + w_ * self.coefs_[v_ceil-1])
                intercepts_.append(self.intercepts_[v_ceil] * (1 - w_) + w_ * self.intercepts_[v_ceil-1])
            else:
                coefs_.append(self.coefs_[0])
                intercepts_.append(self.intercepts_[0])

        if shape == interpolation_grid.shape:
            return np.asarray(coefs_), np.asarray(intercepts_)
        else:
            return np.asarray(coefs_)[0], np.asarray(intercepts_)[0]

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
                              scorers=[]): # GLMScorer instances
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
        """
        check_is_fitted(self, ["coefs_"])

        if alignment not in ['lambda', 'fraction']:
            raise ValueError("alignment must be one of 'lambda' or 'fraction'")

        cloned_path = clone(self)
        if alignment == 'lambda':
            fit_params.update(interpolation_grid=self.lambda_values_)
        else:
            if self.lambda_values is not None:
                warnings.warn('Using pre-specified lambda values, not proportional to lambda_max')
                cloned_path.lambda_values = cloned_path.lambda_values[:self.lambda_values_.shape[0]]
            fit_params = {}

        X, y, groups = indexable(X, y, groups)
        
        cv = check_cv(cv, y, classifier=False)

        predictions = cross_val_predict(cloned_path,
                                        X,
                                        y,
                                        groups=groups,
                                        cv=cv,
                                        n_jobs=n_jobs,
                                        verbose=verbose,
                                        params=fit_params,
                                        pre_dispatch=pre_dispatch)

        # truncate to the size we got
        predictions = predictions[:,:self.lambda_values_.shape[0]]

        response, offset, weight = self.get_data_arrays(X, y, check=False)[2:]

        # adjust for offset
        # because predictions are just X\beta

        if offset is not None:
            predictions = self._offset_predictions(predictions,
                                                   offset)

        splits = [test for _, test in cv.split(np.arange(X.shape[0]))]

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

        self.score_path_ = ScorePath(scores=cv_scores_,
                                     index_best=index_best_,
                                     index_1se=index_1se_,
                                     lambda_values=self.lambda_values_,
                                     norm=np.fabs(self.coefs_).sum(1),
                                     fracdev=self.summary_['Fraction Deviance Explained'],
                                     family=self._family)

        return predictions, self.score_path_
    
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
                               family=self.family,
                               alpha=self.alpha,
                               penalty_factor=self.penalty_factor,
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

    def prefilter(self, X, y):
        """
        Method intended to be overwritten by subclasses to implement pre-filtering of features.
        Allows dynamic computation of an excluded set of features based on X and y.

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
             keep=None):
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
            label = r'Coefficient norms ($\|\beta\|_2$)'
        else:
            label = r'Coefficients ($\beta$)'
        soln_path = pd.DataFrame(coefs_,
                                 columns=self.feature_names,
                                 index=index)
        if drop is not None:
            soln_path = soln_path.drop(columns=drop)
        if keep is not None:
            soln_path = soln_path.loc[:, keep]
        ax = soln_path.plot(ax=ax, legend=False)
        ax.set_xlabel(index.name)
        ax.set_ylabel(label)
        ax.axhline(0, c='k', ls='--')

        if legend:
            fig = ax.figure
            if hasattr(fig, 'get_layout_engine') and fig.get_layout_engine() is not None:
                import warnings
                warnings.warn('If plotting a legend, layout of figure will be set to "constrained".')
            if hasattr(fig, 'set_layout_engine'):
                fig.set_layout_engine('constrained')
            fig.legend(loc='outside right upper')
        return ax
