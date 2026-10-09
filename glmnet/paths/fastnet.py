from copy import copy
import logging
import warnings

from typing import Literal, Optional
from dataclasses import dataclass, field, asdict
   
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ..base import _get_design
from ..glm import GLMState
from ..elnet import (_check_limits,
                    _check_penalty_factor,
                    _design_wrapper_args)
from ..glmnet import (GLMNet,
                      CoefPath)
from ..family import GLMFamilySpec

from .._utils import (_jerr_elnetfit,
                      _validate_cpp_args,
                      _check_offset)

class _NoProgress(object):
    """Stand-in for a tqdm progress bar that shows nothing."""

    def update(self, m):
        pass

    def close(self):
        pass


class _PathProgress(object):
    """
    Progress bar for the C++ paths. These call ``update(m)`` with the
    (0-based) index of the lambda value just fit, as R's ``setpb``,
    rather than an increment.
    """

    def __init__(self, total):
        self.bar = tqdm(total=total)

    def update(self, m):
        self.bar.update(m + 1 - self.bar.n)

    def close(self):
        self.bar.close()


@dataclass
class FastNetControl(object):
    """Control parameters for FastNet path solvers.

    Most fields mirror R's ``glmnet.control``. To change the convergence
    tolerance or iteration limit of the coordinate descent solver, set
    ``thresh`` and ``maxit`` (the analogues of the ``thresh`` and ``maxit``
    arguments to R's ``glmnet``), not ``eps`` or ``mxit``.

    Parameters
    ----------
    fdev : float, default=1e-5
        Minimum fractional change in deviance for stopping the path early.
    eps : float, default=1e-6
        Minimum value of the lambda min ratio; only used when lambda
        values are not supplied. Not a convergence tolerance.
    big : float, default=9.9e35
        Large floating point number, effectively infinity.
    mnlam : int, default=5
        Minimum number of path points (lambda values) fit before early
        stopping is allowed.
    devmax : float, default=0.999
        Path stops early if the fraction of deviance explained reaches
        this value.
    pmin : float, default=1e-9
        Minimum fitted probability for binomial/multinomial models.
    exmx : float, default=250.
        Maximum allowed value of the linear predictor (exponent).
    itrace : int, default=0
        If nonzero, show a progress bar along the path (R's ``trace.it``).
        By default fits are silent.
    prec : float, default=1e-10
        Convergence threshold for the bounds adjustment in multi-response
        (multinomial grouped, multi-Gaussian) fits.
    mxit : int, default=100
        Maximum iterations for the bounds adjustment in multi-response
        (multinomial grouped, multi-Gaussian) fits. Not the coordinate
        descent iteration limit.
    epsnr : float, default=1e-6
        Convergence threshold for Newton-Raphson; kept for parity with
        ``glmnet.control``, not used by the path solvers.
    mxitnr : int, default=25
        Maximum Newton-Raphson iterations; kept for parity with
        ``glmnet.control``, not used by the path solvers.
    maxit : int, default=100000
        Maximum number of passes over the data for coordinate descent,
        across all lambda values.
    thresh : float, default=1e-7
        Convergence threshold for coordinate descent. Each inner loop runs
        until the maximum change in the objective after any coefficient
        update is less than ``thresh`` times the null deviance.
    logging : bool, default=False
        Enable debug logging.
    """

    fdev: float = 1e-5
    eps: float = 1e-6
    big: float = 9.9e35
    mnlam: int = 5
    devmax: float = 0.999
    pmin: float = 1e-9
    exmx: float = 250.
    itrace: int = 0
    prec: float = 1e-10
    mxit: int = 100
    epsnr: float = 1e-6
    mxitnr: int = 25
    # maxit, thresh & logging are not part of glmnet.control
    maxit: int = 100000
    thresh: float = 1e-7
    logging: bool = False
    
@dataclass
class MultiState(object):
    """
    Solution at one lambda value for multiple responses.

    Parameters
    ----------
    coef: np.ndarray
        Coefficients, of shape `(n_features, n_responses)`.
    intercept: np.ndarray
        Intercepts, of shape `(n_responses,)`.
    """
    coef: np.ndarray
    intercept: np.ndarray


@dataclass
class FastNetMixin(GLMNet): # base class for C++ path methods
    """Mixin for fast path solvers using C++ backend.

    This mixin provides the core logic for estimators that use the fast
    C++ implementations of coordinate descent for the elastic net path.

    Parameters
    ----------
    lambda_min_ratio : float, optional
        Minimum lambda ratio.
    nlambda : int, default=100
        Number of lambda values.
    df_max : int, optional
        Maximum degrees of freedom.
    pmax : int, optional
        Maximum number of variables ever nonzero along the path. Defaults
        to `min(2 * df_max + 20, n_features)`, as in R. If it is exceeded,
        the path stops with a warning and the solutions for the larger
        lambdas are returned.
    control : FastNetControl, optional
        Control parameters for the solver.

    Attributes
    ----------
    coefs_ : ndarray
        Fitted coefficients across the path.
    intercepts_ : ndarray
        Fitted intercepts across the path.
    lambda_values_ : ndarray
        The sequence of lambda values used.
    lambda_max_ : float
        The maximum lambda value in the sequence.
    summary_ : pd.DataFrame
        Summary of the fit including Degrees of Freedom and Fraction Deviance Explained.
    """

    lambda_min_ratio: Optional[float] = None
    nlambda: int = 100
    df_max: Optional[int] = None
    pmax: Optional[int] = None
    control: FastNetControl = field(default_factory=FastNetControl)

    # interprets the C++ error code (R's jerr.elnet / jerr.coxnet ...)
    _jerr_message = staticmethod(_jerr_elnetfit)

    def fit(self,
            X,
            y,
            sample_weight=None, # ignored
            interpolation_grid=None):
        """
        Fit the penalized regression model using the FastNet path algorithm.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix.
        y : array-like
            Target vector or matrix.
        sample_weight : array-like, optional
            Sample weights (ignored).
        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.

        Returns
        -------
        self : object
            Fitted estimator.
        """
    
        if not hasattr(self, "_family"):
            self._family = GLMFamilySpec.from_family(self.family, response=y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        self.excluded_ = copy(self.exclude)
        self.excluded_.extend(list(self.prefilter(X, y)))
        self.penalty_factor_ = self.get_penalty_factor(X, y)
        X, y, response, offset, weight = self.get_data_arrays(X, y)

        if not scipy.sparse.issparse(X):
            X = np.asfortranarray(X)
        
        # the C++ path codes handle standardization
        # themselves so we shouldn't handle it at this level
        if not hasattr(self, "design_"):
            if weight is None:
                weight = np.ones(X.shape[0])
            self.design_ = design = _get_design(X,
                                                weight,
                                                standardize=False,
                                                intercept=False)
        else:
            design = self.design_

        if self.df_max is None:
            self.df_max = X.shape[1] + 1
            
        if self.control is None:
            self.control = FastNetControl()

        n_samples, n_features = design.X.shape

        sample_weight = weight
        
        # the C++ paths only advance the bar when itrace is nonzero,
        # so only show one then (R's trace.it)
        if self.control.itrace:
            self.pb = _PathProgress(total=self.nlambda)
        else:
            self.pb = _NoProgress()
        self._args = self._wrapper_args(design,
                                        response,
                                        sample_weight,
                                        offset=offset,
                                        exclude=self.excluded_)

        design_args = _design_wrapper_args(design)
        # 'xm' and 'xs' are used by the elnet / flex CPP code but not the paths
        for k in ['xm', 'xs']:
            if k in design_args:
                del(design_args[k])

        self._args.update(**design_args)
        
        # set control args
        D = asdict(self.control)
        del(D['maxit']) # maxit is not in glmnet.control
        del(D['thresh']) # thresh is not in glmnet.control
        del(D['logging']) # logging is not in glmnet.control
        self._args.update(**D)

        if scipy.sparse.issparse(design.X):
            fit_method = getattr(self, "_sparse", None)

            if fit_method is None:
                raise AttributeError(f"{self.__class__.__name__} has no method '_sparse' required for sparse input.")
        else:
            fit_method = getattr(self, "_dense", None)
            if fit_method is None:
                raise AttributeError(f"{self.__class__.__name__} has no method '_dense' required for dense input.")
        msg = _validate_cpp_args(self._args,
                                  fit_method.__name__)
        if msg is not None:
            raise ValueError(msg)
        self._fit = fit_method(**self._args)
        self.pb.close()

        # the solver fills `ca` in place; the C++ wrappers don't return it
        # because pybind11 would copy it
        self._fit['ca'] = self._args.pop('ca')

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if self._fit['jerr'] != 0:
            errmsg = type(self)._jerr_message(self._fit['jerr'],
                                              self.control.maxit,
                                              pmax=self._args['nx'])
            if self.control.logging: logging.debug(errmsg['msg'])
            if not errmsg['fatal']:
                # as R's glmnet, warn that solutions for larger lambdas were returned
                warnings.warn(errmsg['msg'])

        # extract the coefficients
        
        result = self._extract_fits(X.shape, response.shape)
        n_features = design.X.shape[1]

        self.coefs_ = result['coefs']
        self.intercepts_ = result['intercepts']
            
        # single response: coefs_ has shape (nlambda, nfeatures)
        if self.coefs_.ndim == 2:
            self.state_ = GLMState(self.coefs_[-1],
                                   self.intercepts_[-1])
        # multiple responses: coefs_ has shape (nlambda, nfeatures, nresponse)
        elif self.coefs_.ndim == 3:
            self.state_ = MultiState(self.coefs_[-1],
                                     self.intercepts_[-1])

        self.lambda_values_ = result['lambda_values']
        nfits = self.lambda_values_.shape[0]
        dev_ratios_ = self._fit['dev'][:nfits]
        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = result['df']
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        # set lambda_max

        # following https://github.com/trevorhastie/glmnet/blob/master/R/glmnet.R#L523
        # will work with equispaced values on log scale

        if len(self.lambda_values_) > 2 and self.lambda_values is None:
            self.lambda_values_[0] = self.lambda_values_[1]**2 / self.lambda_values_[2]

        self.lambda_max_ = self.lambda_values_[0]

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

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape): # getcoef.R
        """
        Extract fitted coefficients, intercepts, and related statistics.

        Parameters
        ----------
        X_shape : tuple
            Shape of the input feature matrix.
        response_shape : tuple
            Shape of the response array.

        Returns
        -------
        dict
            Dictionary with keys 'coefs', 'intercepts', 'df', and 'lambda_values'.

        Notes
        -----
        When the solver's ``ca`` buffer has a column per feature
        (``nx == n_features``), ``coefs`` is built in place in it rather
        than in a second ``(nfits, n_features)`` array.
        """
        _fit, _args = self._fit, self._args
        n_features = X_shape[1]
        nfits = _fit['lmu']
        nx = _args['nx']

        if nfits < 1:
            # as in R's getcoef: a single all-zero fit at lambda = Inf
            warnings.warn("an empty model has been returned; probably a convergence issue")
            return {'coefs':np.zeros((1, n_features)),
                    'intercepts':np.asarray(_fit['a0']).reshape(-1)[:1],
                    'df':np.zeros(1, dtype=int),
                    'lambda_values':np.array([np.inf])}

        nin = _fit['nin'][:nfits]
        ninmax = int(nin.max()) if nin.size else 0
        lambda_values = _fit['alm'][:nfits]
        intercepts = _fit['a0'][:nfits]

        ca = _fit.pop('ca') # `coefs` may be built in place in it, see below

        if ninmax > 0:
            if ca.ndim == 1: # logistic is like this: one block of nx per lambda
                unsort_coefs = ca[:(nx*nfits)].reshape(nfits, nx)
            else: # (nx, nlambda) Fortran-ordered, so .T is a C-ordered view
                unsort_coefs = ca[:,:nfits].T

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1
            active = unsort_coefs[:, :ninmax]
            df = (np.fabs(active) > 0).sum(1)

            if nx == n_features:
                # scatter in place instead of allocating a second copy
                active = active.copy()
                unsort_coefs[:] = 0
                unsort_coefs[:, active_seq] = active
                coefs = unsort_coefs
            else: # df_max shrank the buffer, it can't hold the dense result
                coefs = np.zeros((nfits, n_features))
                coefs[:, active_seq] = active
        else:
            # No features selected; return zeros
            coefs = np.zeros((nfits, n_features))
            df = np.zeros(nfits, dtype=int)

        return {'coefs':coefs,
                'intercepts':intercepts,
                'df':df,
                'lambda_values':lambda_values}
 
    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset, # ignored, but subclasses use it
                      exclude=[]):
        """
        Prepare arguments for the C++ backend wrapper.

        Parameters
        ----------
        design : object
            Design matrix and related info.
        response : array-like
            Response array.
        sample_weight : array-like
            Sample weights.
        offset : array-like
            Offset array (ignored here).
        exclude : list, optional
            Indices to exclude from penalization.

        Returns
        -------
        dict
            Arguments for the backend solver.
        """

        if self.lambda_values is not None:
            self.lambda_values = np.asarray(self.lambda_values)
            
        sample_weight = np.asfortranarray(sample_weight)
        
        X = design.X
        n_samples, n_features = X.shape

        if self.lambda_min_ratio is None:
            if n_samples < n_features:
                self.lambda_min_ratio = 1e-2
            else:
                self.lambda_min_ratio = 1e-4

        if self.lambda_values is None:
            if self.lambda_min_ratio > 1:
                raise ValueError('lambda_min_ratio should be less than 1')
            flmin = float(self.lambda_min_ratio)
            ulam = np.zeros((1, 1))
        else:
            flmin = 1.
            # Convert to array if it's a list
            self.lambda_values = np.asarray(self.lambda_values)
            if np.any(self.lambda_values < 0):
                raise ValueError('lambdas should be non-negative')
            ulam = np.asfortranarray(np.sort(self.lambda_values)[::-1].reshape((-1, 1)))
            self.nlambda = self.lambda_values.shape[0]

        if response.ndim == 1:
            response = response.reshape((-1,1))

        # compute vp
        penalty_factor_, excluded_ = _check_penalty_factor(self.penalty_factor_,
                                                                n_features,
                                                                exclude)
        self.excluded_ = np.asarray(excluded_) - 1

        # compute jd
        # assume that there are no constant variables

        if len(excluded_) > 0:
            jd = np.hstack([len(excluded_), excluded_]).astype(np.int32)
        else:
            jd = np.array([0], np.int32)
            
        lower_limits_, upper_limits_ = _check_limits(self.lower_limits,
                                                     self.upper_limits,
                                                     n_features,
                                                     big=self.control.big)

        # compute cl from upper and lower limits

        if not np.all(lower_limits_ <= 0):
            raise ValueError('lower limits should be <= 0')

        if not np.all(upper_limits_ >= 0):
            raise ValueError('upper limits should be >= 0')

        cl = np.asarray([lower_limits_,
                         upper_limits_], float)
        
        if np.any(cl[0] == 0) or np.any(cl[-1] == 0):
            self.control.fdev = 0

        # all but the X -- this is set below

        # nx is R's pmax: the maximum number of variables ever nonzero
        if self.pmax is not None:
            if int(self.pmax) != self.pmax or self.pmax < 1:
                raise ValueError('pmax should be a positive integer')
            nx = int(self.pmax)
        elif self.df_max is not None:
            nx = min(self.df_max*2+20, n_features)
        else:
            nx = n_features

        _args = {'parm':float(self.alpha),
                 'ni':n_features,
                 'no':n_samples,
                 'y':np.asfortranarray(response),
                 'w': np.asarray(sample_weight).reshape((-1, 1)),
                 'jd': jd,
                 'vp': np.asarray(penalty_factor_).reshape((-1, 1)),
                 'cl': np.asfortranarray(cl),
                 'ne': self.df_max,
                 'nx': nx,
                 'nlam': self.nlambda,
                 'flmin':flmin,
                 'ulam':ulam,
                 'thr':float(self.control.thresh),
                 'isd':int(self.standardize),
                 'intr':int(self.fit_intercept),
                 'maxit':int(self.control.maxit),
                 'pb':self.pb,
                 'lmu':0, # these asfortran calls not necessary -- nullop
                 'a0':np.asfortranarray(np.zeros((self.nlambda, 1), float)),
                 'ca':np.zeros((nx, self.nlambda), order='F'),
                 'ia':np.zeros((nx, 1), np.int32),
                 'nin':np.zeros((self.nlambda, 1), np.int32),
                 'nulldev':0.,
                 'dev':np.zeros((self.nlambda, 1)),
                 'alm':np.zeros((self.nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        return _args

    def _fixed_lambda_family(self):
        # the `family` field is unused by the C++ paths; `_family` is set in
        # __post_init__ (e.g. binomial for LogNet) or by `fit`
        return self._family

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
class MultiFastNetMixin(FastNetMixin): # paths with multiple responses
    """
    Mixin for fast path solvers with multiple response variables.

    Provides common functionality for models predicting multiple responses
    such as multi-response Gaussian regression and multinomial classification.

    Parameters
    ----------
    lambda_min_ratio : float, optional
        Minimum lambda ratio.
    nlambda : int, default=100
        Number of lambda values.
    df_max : int, optional
        Maximum degrees of freedom.
    control : FastNetControl, optional
        Control parameters for the solver.
    """

    def get_fixed_lambda(self,
                         lambda_val):
        """
        Get an estimator for a fixed lambda value.

        There is no single lambda solver for multiple responses, so the
        estimator fits this path's lambda values above `lambda_val`, then
        `lambda_val` itself, and keeps the last solution: the path supplies
        the warm starts, as for R's ``coef(..., exact=TRUE)``.

        Parameters
        ----------
        lambda_val: float
            Lambda value.

        Returns
        -------
        tuple
            (estimator, state) where estimator is a `FixedLambdaMultiNet`
            and state is a `MultiState` with the coefficients interpolated
            from the path at `lambda_val`.
        """
        check_is_fitted(self, ["coefs_", "feature_names_in_"])

        if lambda_val < 0:
            raise ValueError('lambda_val must be non-negative')
        lambda_values = self.lambda_values_[self.lambda_values_ > lambda_val]
        estimator = FixedLambdaMultiNet(path_estimator=clone(self),
                                        lambda_val=lambda_val,
                                        lambda_values=np.hstack([lambda_values, lambda_val]))

        coefs, intercepts = self.interpolate_coefs([lambda_val])
        state = MultiState(coefs[0], intercepts[0])
        return estimator, state

    def predict(self,
                X,
                prediction_type='link', # ignored except checking valid
                interpolation_grid=None,
                offset=None,
                ):
        """
        Predict using the fitted model for multiple responses.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        prediction_type : str, optional
            Type of prediction ('response' or 'link').
        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.
        offset : array-like, optional
            Offset for the rows of `X`, of shape `(n_samples, n_responses)`,
            added to the linear predictor (R's `newoffset`). A vector is used
            for every response. If omitted, no offset is used.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        if interpolation_grid is not None:
            grid_ = np.asarray(interpolation_grid)
            squeeze = grid_.ndim == 0
            grid_ = np.atleast_1d(grid_)
            coefs_, intercepts_ = self.interpolate_coefs(grid_)
        else:
            grid_ = None
            squeeze = False
            coefs_, intercepts_ = self.coefs_, self.intercepts_

            
        if prediction_type not in ['response', 'link']:
            raise ValueError("prediction should be one of 'response' or 'link'")
        
        term1 = np.einsum('ijk,lj->ilk',
                          coefs_,
                          X)
        fits = term1 + intercepts_[:, None, :]
        fits = np.transpose(fits, [1,0,2])
        if offset is not None:
            fits = fits + _check_offset(offset, X.shape[0], fits.shape[2])[:, None, :]

        # make return based on original
        # promised number of lambdas
        # pad with last value

        # if possible we might want to do less than `self.nlambda`
        
        if interpolation_grid is not None:
            nlambda = coefs_.shape[0]
        else:
            nlambda = self.nlambda

        value = np.empty((fits.shape[0],
                          nlambda,
                          fits.shape[2]), float) * np.nan
        value[:,:fits.shape[1]] = fits
        value[:,fits.shape[1]:] = fits[:,-1][:,None]

        if not squeeze:
            return value
        else:
            return value[:,0,:]

    def nonzero(self,
                interpolation_grid=None):
        """
        Indices of the nonzero coefficients along the path, as
        `predict(fit, type="nonzero")` in R.

        Parameters
        ----------
        interpolation_grid : array-like, optional
            Grid of lambda values. If provided, coefficients are interpolated
            to these values first, as in `predict`.

        Returns
        -------
        list
            For each lambda in `lambda_values_` (or in `interpolation_grid`),
            the (0-based) indices of the features with a nonzero coefficient
            for any response. For an ungrouped multinomial fit (`grouped=False`)
            this is instead a list with one such list per class, as in R.
            If `interpolation_grid` is a scalar, each list of arrays is
            replaced by its single array.
        """
        coefs_, squeeze = self._nonzero_coefs(interpolation_grid)
        # coefs_ has shape (n_lambda, n_features, n_responses);
        # MultiGaussNet has no `grouped` attribute and is always grouped
        if getattr(self, 'grouped', True):
            value = [np.nonzero(np.any(c != 0, axis=1))[0] for c in coefs_]
            return value[0] if squeeze else value
        value = [[np.nonzero(c[:, k])[0] for c in coefs_]
                 for k in range(coefs_.shape[2])]
        return [v[0] for v in value] if squeeze else value

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        """
        Extract fitted coefficients, intercepts, and related statistics for multi-response models.

        Parameters
        ----------
        X_shape : tuple
            Shape of the input feature matrix.
        response_shape : tuple
            Shape of the response array.

        Returns
        -------
        dict
            Dictionary with keys 'coefs', 'intercepts', 'df', and 'lambda_values'.
        """
        _fit, _args = self._fit, self._args
        n_features = X_shape[1]
        nresp = response_shape[1]
        nfits = _fit['lmu']
        if nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        nin = _fit['nin'][:nfits]
        ninmax = int(nin.max()) if nin.size else 0
        lambda_values = _fit['alm'][:nfits]
        intercepts = _fit['a0'][:,:nfits].T
        ca = _fit.pop('ca')

        if ninmax > 0:
            # flattened (nx, nresp, nlam) column-major, as in R's getcoef.multinomial
            nx = _args['nx']
            unsort_coefs = ca[:(nresp*nx*nfits)].reshape(nfits,
                                                         nresp,
                                                         nx)
            unsort_coefs = np.transpose(unsort_coefs, [0,2,1])
            df = ((unsort_coefs**2).sum(2) > 0).sum(1)

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1

            coefs = np.zeros((nfits, n_features, nresp))
            coefs[:, active_seq] = unsort_coefs[:, :len(active_seq)]
        else:
            # No features selected; return zeros
            coefs = np.zeros((nfits, n_features, nresp))
            df = np.zeros(nfits, dtype=int)

        return {'coefs':coefs,
                'intercepts':intercepts,
                'df':df,
                'lambda_values':lambda_values}


    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """
        Prepare arguments for the C++ backend wrapper for multi-response models.

        Parameters
        ----------
        design : object
            Design matrix and related info.
        response : array-like
            Response array.
        sample_weight : array-like
            Sample weights.
        offset : array-like
            Offset array.
        exclude : list, optional
            Indices to exclude from penalization.

        Returns
        -------
        dict
            Arguments for the backend solver.
        """
        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # ensure shapes are correct

        (n_samples, n_features), nr = design.X.shape, response.shape[1]
        _args['a0'] = np.asfortranarray(np.zeros((nr, self.nlambda), float))
        _args['ca'] = np.zeros(self.nlambda * nr * _args['nx'])
        _args['y'] = np.asfortranarray(_args['y'].reshape((n_samples, nr)))

        return _args


@dataclass
class FixedLambdaMultiNet(BaseEstimator):
    """
    Fit of a multiple response path estimator at one lambda value, as
    returned by `MultiFastNetMixin.get_fixed_lambda`.

    The path estimator is fit to `lambda_values`, which end at `lambda_val`,
    and the solution at `lambda_val` is kept.

    Parameters
    ----------
    path_estimator: MultiFastNetMixin
        Unfitted path estimator (e.g. `MultiGaussNet` or `MultiClassNet`).
    lambda_val: float
        Lambda value.
    lambda_values: np.ndarray
        Decreasing lambda values of the path fit, ending at `lambda_val`.

    Attributes
    ----------
    coef_: np.ndarray
        Coefficients at `lambda_val`, of shape `(n_features, n_responses)`.
    intercept_: np.ndarray
        Intercepts at `lambda_val`, of shape `(n_responses,)`.
    path_: MultiFastNetMixin
        The fitted path estimator.
    """
    path_estimator: BaseEstimator
    lambda_val: float
    lambda_values: np.ndarray

    def fit(self,
            X,
            y,
            sample_weight=None,  # ignored
            warm_state=None):
        """
        Fit at `lambda_val`.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix.
        y : array-like
            Target matrix, with any weight or offset columns.
        sample_weight : array-like, optional
            Sample weights (ignored).
        warm_state : MultiState, optional
            Ignored: the C++ paths take no warm start, the path from the
            largest lambda value supplies it.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        path = clone(self.path_estimator)
        path.lambda_values = np.asarray(self.lambda_values, float)
        path.fit(X, y)
        if not np.isclose(path.lambda_values_[-1], self.lambda_val):
            warnings.warn('the path stopped before reaching lambda_val; '
                          'returning the solution at the smallest lambda fitted')
        self.path_ = path
        self.coef_ = path.coefs_[-1]
        self.intercept_ = path.intercepts_[-1]
        self.state_ = MultiState(self.coef_, self.intercept_)
        return self

    def predict(self,
                X,
                prediction_type=None):
        """
        Predict at `lambda_val`.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        prediction_type : str, optional
            As for the path estimator's `predict`, whose default is used
            if None.

        Returns
        -------
        np.ndarray
            Predictions, of shape `(n_samples, n_responses)` (or
            `(n_samples,)` for `prediction_type='class'`).
        """
        check_is_fitted(self, ["coef_"])
        kwargs = {} if prediction_type is None else {'prediction_type': prediction_type}
        return self.path_.predict(X, **kwargs)[:, len(self.path_.lambda_values_) - 1]
