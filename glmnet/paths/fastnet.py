from copy import copy
import logging
import warnings

from typing import Literal, Optional
from dataclasses import dataclass, field, asdict
   
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from ..base import _get_design
from ..glm import GLMState
from ..elnet import (_check_limits,
                    _check_penalty_factor,
                    _design_wrapper_args)
from ..glmnet import (GLMNet,
                      CoefPath)
from ..family import GLMFamilySpec

from .._utils import (_jerr_elnetfit,
                      _validate_cpp_args)

@dataclass
class FastNetControl(object):
    """Control parameters for FastNet path solvers.

    Parameters
    ----------
    fdev : float, default=1e-5
        Fractional deviance tolerance for early stopping.
    eps : float, default=1e-6
        Convergence threshold for coordinate descent.
    big : float, default=9.9e35
        Large value used for numerical stability.
    mnlam : int, default=5
        Minimum number of lambda values.
    devmax : float, default=0.999
        Maximum fraction of deviance explained.
    pmin : float, default=1e-9
        Minimum value for probabilities.
    exmx : float, default=250.
        Maximum exponent value.
    itrace : int, default=0
        Trace level for logging.
    prec : float, default=1e-10
        Precision for calculations.
    mxit : int, default=100
        Maximum number of iterations.
    epsnr : float, default=1e-6
        Convergence threshold for Newton-Raphson.
    mxitnr : int, default=25
        Maximum Newton-Raphson iterations.
    maxit : int, default=100000
        Maximum number of iterations (wrapper only).
    thresh : float, default=1e-7
        Threshold for convergence (wrapper only).
    logging : bool, default=False
        Enable logging (wrapper only).
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
    # thresh & logging not part of glmnet.control but used in the wrapper
    maxit: int = 100000
    thresh: float = 1e-7
    logging: bool = False
    
@dataclass
class FastNetMixin(GLMNet): # base class for C++ path methods
    """Mixin for fast path solvers using C++ backend.

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

    lambda_min_ratio: Optional[float] = None
    nlambda: int = 100
    df_max: Optional[int] = None
    control: FastNetControl = field(default_factory=FastNetControl)

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
        
        self.pb = tqdm(total=self.nlambda)
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

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if self._fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(self._fit['jerr'], self.control.maxit)
            if self.control.logging: logging.debug(errmsg['msg'])

        # extract the coefficients
        
        result = self._extract_fits(X.shape, response.shape)
        n_features = design.X.shape[1]

        self.coefs_ = result['coefs']
        self.intercepts_ = result['intercepts']
            
        if self.coefs_.ndim == 1:
            self.state_ = GLMState(self.coefs_[-1],
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
        """
        _fit, _args = self._fit, self._args
        n_features = X_shape[1]
        nfits = _fit['lmu']

        if nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        nin = _fit['nin'][:nfits]
        ninmax = max(nin)
        lambda_values = _fit['alm'][:nfits]

        if ninmax > 0:
            if _fit['ca'].ndim == 1: # logistic is like this
                unsort_coefs = _fit['ca'][:(n_features*nfits)].reshape(nfits, n_features)
            else:
                unsort_coefs = _fit['ca'][:,:nfits].T
            df = (np.fabs(unsort_coefs) > 0).sum(1)

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1

            coefs = np.zeros((nfits, n_features))
            coefs[:, active_seq] = unsort_coefs[:, :len(active_seq)]
            intercepts = _fit['a0'][:nfits]

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
        penalty_factor_, excluded_ = _check_penalty_factor(self.penalty_factor,
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

        # isn't this always n_features?
        # should have a df_max arg
        if self.df_max is not None:
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
                 'ca':np.asfortranarray(np.zeros((nx, self.nlambda))),
                 'ia':np.zeros((nx, 1), np.int32),
                 'nin':np.zeros((self.nlambda, 1), np.int32),
                 'nulldev':0.,
                 'dev':np.zeros((self.nlambda, 1)),
                 'alm':np.zeros((self.nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        return _args

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
    """

    def predict(self,
                X,
                prediction_type='link', # ignored except checking valid
                interpolation_grid=None,
                ):
        """
        Predict using the fitted model for multiple responses.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        prediction_type : str, optional
            Type of prediction ('response' or 'link').

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
        ninmax = max(nin)
        lambda_values = _fit['alm'][:nfits]

        if ninmax > 0:
            unsort_coefs = _fit['ca'][:(nresp*n_features*nfits)].reshape(nfits,
                                                                    nresp,
                                                                    n_features)
            unsort_coefs = np.transpose(unsort_coefs, [0,2,1])
            df = ((unsort_coefs**2).sum(2) > 0).sum(1)

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1

            coefs = np.zeros((nfits, n_features, nresp))
            coefs[:, active_seq] = unsort_coefs[:, :len(active_seq)]
            intercepts = _fit['a0'][:,:nfits].T

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
        _args['ca'] = np.zeros((self.nlambda * nr * n_features, 1))
        _args['y'] = np.asfortranarray(_args['y'].reshape((n_samples, nr)))

        return _args
