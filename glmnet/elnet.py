import logging

from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y

from ._elnet_point import elnet_point as dense_wls
from ._elnet_point import spelnet_point as sparse_wls
from .base import (Base,
                   Penalty,
                   Design,
                   _get_design)
from ._utils import _jerr_elnetfit

@dataclass
class ElNetControl(object):
    """Control parameters for ElNet optimization.
    
    Parameters
    ----------
    thresh : float, default=1e-7
        Convergence threshold for optimization.
    maxit : int, default=100000
        Maximum number of iterations.
    big : float, default=9.9e35
        Large number used for bounds.
    logging : bool, default=False
        Whether to enable debug logging.
    """
    thresh: float = 1e-7
    maxit: int = 100000
    big: float = 9.9e35
    logging: bool = False

@dataclass
class ElNetSpec(Penalty):
    """Specification for ElNet model parameters.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    standardize : bool, default=True
        Whether to standardize features.
    control : ElNetControl, optional
        Control parameters for optimization.
    exclude : list, default_factory=list
        List of variable indices to exclude from penalization.
    """
    fit_intercept: bool = True
    standardize: bool = True
    control: ElNetControl = field(default_factory=ElNetControl)
    exclude: list = field(default_factory=list)

@dataclass
class ElNet(BaseEstimator,
            RegressorMixin,
            ElNetSpec):
    """Elastic Net regression model.
    
    This class implements elastic net regression using coordinate descent.
    It supports both dense and sparse input matrices.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing parameter. alpha=1 is lasso, alpha=0 is ridge.
    lambda_val : float, default=0.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    standardize : bool, default=True
        Whether to standardize features.
    penalty_factor : array-like, optional
        Multiplicative factors for penalty for each coefficient.
    lower_limits : array-like, optional
        Lower bounds for coefficients.
    upper_limits : array-like, optional
        Upper bounds for coefficients.
    control : ElNetControl, optional
        Control parameters for optimization.
    exclude : list, default_factory=list
        List of variable indices to exclude from penalization.
    """
    def fit(self,
            X,
            y,
            sample_weight=None,
            warm=None,
            check=True):
        """Fit the elastic net model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        warm : tuple, optional
            Warm start parameters (coef, intercept, eta).
        check : bool, default=True
            Whether to perform input validation.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(self, "design_"):
            self.design_ = design = _get_design(X,
                                                sample_weight,
                                                standardize=self.standardize,
                                                intercept=self.fit_intercept)
        else:
            design = self.design_

        if self.lambda_val > 0 or not (np.all(design.centers_ == 0) and np.all(design.scaling_ == 1)):
            if self.control is None:
                self.control = ElNetControl()
            n_samples, n_features = design.X.shape
            design.X, y = check_X_y(design.X, y,
                                    accept_sparse=['csc'],
                                    multi_output=False,
                                    estimator=self)
            if sample_weight is None:
                sample_weight = np.ones(n_samples) / n_samples
            lower_limits_, upper_limits_ = _check_limits(self.lower_limits,
                                                         self.upper_limits,
                                                         n_features,
                                                         big=self.control.big)
            penalty_factor_, excluded_ = _check_penalty_factor(self.penalty_factor,
                                                               n_features,
                                                               self.exclude)
            self.excluded_ = np.asarray(excluded_) - 1

            if hasattr(self, "_wrapper_args") and warm is not None:
                args = self._wrapper_args
                nulldev = self._nulldev
                coef, intercept, eta = warm
                if ((design.shape[1] != coef.shape[0] + 1) or
                    (design.shape[0] != eta.shape[0])): 
                    raise ValueError('dimension mismatch for warm start and design')
                args['aint'] = intercept
                args['a'][:] = coef.reshape((-1,1))
                args['r'][:] = (sample_weight * (y - eta)).reshape((-1,1))
            else:
                args, nulldev = _elnet_wrapper_args(design,
                                                    y,
                                                    sample_weight,
                                                    self.lambda_val,
                                                    alpha=self.alpha,
                                                    intercept=self.fit_intercept,
                                                    penalty_factor=penalty_factor_,
                                                    exclude=excluded_,
                                                    lower_limits=lower_limits_,
                                                    upper_limits=upper_limits_,
                                                    thresh=self.control.thresh,
                                                    maxit=self.control.maxit)
                self._wrapper_args = args
                self._nulldev = nulldev
            args['alpha'] = float(self.alpha)
            args['almc'] = float(self.lambda_val)
            args['intr'] = int(self.fit_intercept)
            args['jerr'] = 0
            args['maxit'] = int(self.control.maxit)
            args['thr'] = float(self.control.thresh)
            args['v'] = np.asarray(sample_weight, float).reshape((-1,1))
            if self.control.logging: logging.debug(f'Elnet warm coef: {args["a"]}, Elnet warm intercept: {args["aint"]}')
            check_resid = True
            if check_resid:
                r1 = args['r'].reshape(-1)
                r2 = sample_weight * (y - design @ np.hstack([args['aint'], args['a'].reshape(-1)])).reshape(-1)
                if np.linalg.norm(r1-r2) / max(np.linalg.norm(r1), 1) > 1e-6:
                    raise ValueError('resid not set correctly')
            if scipy.sparse.issparse(design.X):
                wls_fit = sparse_wls(**args)
            else:
                wls_fit = dense_wls(**args)
            if wls_fit['jerr'] != 0:
                errmsg = _jerr_elnetfit(wls_fit['jerr'], self.control.maxit)
                if self.control.logging: logging.debug(errmsg['msg'])
            if self.control.logging: logging.debug(f'Elnet coef: {wls_fit["a"]}, Elnet intercept: {wls_fit["aint"]}')
            self.raw_coef_ = wls_fit['a'].reshape(-1)
            self.raw_intercept_ = wls_fit['aint']
        else:
            lm = LinearRegression(fit_intercept=self.fit_intercept)
            if scipy.sparse.issparse(design.X):
                X_s = scipy.sparse.csc_array(design.X)
            else:
                X_s = design.X
            lm.fit(X_s, y, sample_weight)
            self.raw_coef_ = lm.coef_
            self.raw_intercept_ = lm.intercept_
        self.design_ = design
        self.coef_ = self.raw_coef_ / design.scaling_
        self.intercept_ = self.raw_intercept_ - (self.coef_ * self.design_.centers_).sum()
        return self

def _elnet_wrapper_args(design,
                        y,
                        sample_weight,
                        lambda_val,
                        alpha=1.0,
                        intercept=True,
                        thresh=1e-7,
                        maxit=100000,
                        penalty_factor=None, 
                        exclude=[],
                        lower_limits=-np.inf,
                        upper_limits=np.inf):
    """Create wrapper arguments for ElNet C++ function.
    
    Parameters
    ----------
    design : Design
        Design matrix object.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Sample weights.
    lambda_val : float
        Regularization strength.
    alpha : float, default=1.0
        Elastic net mixing parameter.
    intercept : bool, default=True
        Whether to fit intercept.
    thresh : float, default=1e-7
        Convergence threshold.
    maxit : int, default=100000
        Maximum iterations.
    penalty_factor : array-like, optional
        Penalty factors for each coefficient.
    exclude : list, default=[]
        Variables to exclude from penalization.
    lower_limits : array-like, optional
        Lower bounds for coefficients.
    upper_limits : array-like, optional
        Upper bounds for coefficients.
        
    Returns
    -------
    args : dict
        Arguments for C++ function.
    nulldev : float
        Null deviance.
    """
    X = design.X
    exclude = np.asarray(exclude, np.int32)
    n_samples, n_features = X.shape
    if penalty_factor is None:
        penalty_factor = np.ones(n_features)
    ybar = np.sum(y * sample_weight) / np.sum(sample_weight)
    nulldev = np.sum(sample_weight * (y - ybar)**2)
    ju = np.ones((n_features, 1), np.int32)
    ju[exclude] = 0
    cl = np.asfortranarray([lower_limits,
                            upper_limits], float)
    nx = n_features
    a  = np.zeros((n_features, 1))
    aint = 0.
    alm0  = 0.
    g = np.zeros((n_features, 1))
    ia = np.zeros((nx, 1), np.int32)
    iy = np.zeros((n_features, 1), np.int32)
    iz = 0
    m = 1
    mm = np.zeros((n_features, 1), np.int32)
    nino = int(0)
    nlp = 0
    r =  (sample_weight * y).reshape((-1,1))
    rsqc = 0.
    xv = np.zeros((n_features, 1))
    alpha = float(alpha)
    almc = float(lambda_val)
    intr = int(intercept)
    jerr = 0
    maxit = int(maxit)
    thr = float(thresh)
    v = np.asarray(sample_weight, float).reshape((-1,1))
    a_new = a
    _args = {'alm0':alm0,
             'almc':almc,
             'alpha':alpha,
             'm':m,
             'no':n_samples,
             'ni':n_features,
             'r':r,
             'xv':xv,
             'v':v,
             'intr':intr,
             'ju':ju,
             'vp':penalty_factor,
             'cl':cl,
             'nx':nx,
             'thr':thr,
             'maxit':maxit,
             'a':a,
             'aint':aint,
             'g':g,
             'ia':ia,
             'iy':iy,
             'iz':iz,
             'mm':mm,
             'nino':nino,
             'rsqc':rsqc,
             'nlp':nlp,
             'jerr':jerr}
    _args.update(**_design_wrapper_args(design))
    return _args, nulldev

def _check_limits(lower_limits,
                  upper_limits,
                  n_features,
                  big=9.9e35):
    """
    Validate and broadcast lower and upper coefficient limits for elastic net models.

    Parameters
    ----------
    lower_limits : float or array-like
        Lower bounds for coefficients. Can be a scalar or array of length n_features.
    upper_limits : float or array-like
        Upper bounds for coefficients. Can be a scalar or array of length n_features.
    n_features : int
        Number of features.
    big : float, default=9.9e35
        Value to use in place of infinite bounds.

    Returns
    -------
    lower_limits : np.ndarray
        Array of lower bounds, shape (n_features,).
    upper_limits : np.ndarray
        Array of upper bounds, shape (n_features,).

    Raises
    ------
    ValueError
        If the provided limits are not compatible with n_features.
    """
    lower_limits = np.asarray(lower_limits)
    upper_limits = np.asarray(upper_limits)

    if lower_limits.shape in [(), (1,)]:
        lower_limits = lower_limits * np.ones(n_features)
    if upper_limits.shape in [(), (1,)]:
        upper_limits = upper_limits * np.ones(n_features)
    lower_limits = lower_limits[:n_features]
    upper_limits = upper_limits[:n_features]
    if lower_limits.shape[0] < n_features:
        raise ValueError('lower_limits should have shape {0}, but has shape {1}'.format((n_features,),
                                                                                        lower_limits.shape))
    if upper_limits.shape[0] < n_features:
        raise ValueError('upper_limits should have shape {0}, but has shape {1}'.format((n_features,),
                                                                                        upper_limits.shape))
    lower_limits[lower_limits == -np.inf] = -big
    upper_limits[upper_limits == np.inf] = big
    return lower_limits, upper_limits

def _check_penalty_factor(penalty_factor,
                          n_features,
                          exclude):
    """
    Validate and broadcast penalty factors for elastic net models, and update excluded variables.

    Parameters
    ----------
    penalty_factor : float or array-like or None
        Penalty factors for each coefficient. If None, defaults to ones. Infinite values indicate exclusion.
    n_features : int
        Number of features.
    exclude : list
        List of variable indices to exclude from penalization (0-based).

    Returns
    -------
    vp : np.ndarray
        Normalized penalty factors, shape (n_features, 1).
    exclude : list
        Updated list of excluded variable indices (1-based for C++ backend).

    Raises
    ------
    ValueError
        If any excluded variable index is out of range.
    """

    if penalty_factor is None:
        penalty_factor = np.ones(n_features)
    else:
        penalty_factor = np.asarray(penalty_factor)
    _isinf_penalty = np.isinf(penalty_factor)
    if np.any(_isinf_penalty):
        exclude.extend(np.nonzero(_isinf_penalty)[0])
        exclude = np.unique(exclude)
    exclude = list(np.asarray(exclude, int))
    if len(exclude) > 0:
        if max(exclude) >= n_features:
            raise ValueError("Some excluded variables out of range")
        penalty_factor[exclude] = 1
    vp = np.maximum(0, penalty_factor).reshape((-1,1))
    vp = (vp * n_features / vp.sum())
    exclude = list(np.asarray(exclude, int) + 1)
    return vp, list(exclude)

def _design_wrapper_args(design):
    """Create design wrapper arguments for C++ function.
    
    Parameters
    ----------
    design : Design
        Design matrix object.
        
    Returns
    -------
    dict
        Arguments for C++ function.
    """
    if not scipy.sparse.issparse(design.X):
        return {'x':design.X}
    else:
        return {'x_data_array':design.X.data,
                'x_indices_array':design.X.indices,
                'x_indptr_array':design.X.indptr,
                'xm':design.centers_,
                'xs':design.scaling_}
