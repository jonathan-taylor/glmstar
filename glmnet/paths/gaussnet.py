import logging
import warnings

from typing import Literal, Optional
from dataclasses import dataclass, field
   
import numpy as np
import pandas as pd
import scipy.sparse

from .fastnet import FastNetMixin

from .._gaussnet import gaussnet as _dense
from .._gaussnet import spgaussnet as _sparse

"""
Implements the GaussNet path algorithm for Gaussian regression models.
Provides the GaussNet estimator class using the FastNetMixin base.
"""

@dataclass
class GaussNet(FastNetMixin):
    """GaussNet estimator for Gaussian regression using the FastNet path algorithm.

    This class implements the regularization path for Gaussian (linear) regression
    models using coordinate descent.

    Parameters
    ----------
    covariance : bool or None, optional
        Whether to use the covariance update (default: auto based on nvars).
    lambda_min_ratio : float, optional
        Minimum lambda ratio.
    nlambda : int, default=100
        Number of lambda values.
    df_max : int, optional
        Maximum degrees of freedom.
    control : FastNetControl, optional
        Control parameters for the solver.

    Attributes
    ----------
    coefs_ : ndarray of shape (n_lambda, n_features)
        Fitted coefficients across the path.
    intercepts_ : ndarray of shape (n_lambda,)
        Fitted intercepts across the path.
    lambda_values_ : ndarray of shape (n_lambda,)
        The sequence of lambda values used.
    response_scale_ : float
        The scale of the response, its weighted standard deviation (root
        mean square without an intercept, after any offset). As R's
        ``glmnet(family="gaussian")``, the path fits the response divided
        by it, so that for `alpha < 1` the ridge part of the penalty is
        `lambda * (1 - alpha) / response_scale_`. The IRLS `GLMNet` and
        `RegGLM` with a Gaussian family (R's ``family=gaussian()``) do not
        rescale: see `get_fixed_lambda`.
    """

    covariance : Optional[bool] = None

    _dense = _dense
    _sparse = _sparse

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        """Extract fitted coefficients, intercepts, and related statistics for Gaussian models.

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
        self._fit['dev'] = self._fit['rsq'] # gaussian fit calls it rsq
        return super()._extract_fits(X_shape,
                                     response_shape)
        
    def get_fixed_lambda(self,
                         lambda_val):
        """
        Get a regularized estimator for a fixed lambda value.

        GaussNet, as R's ``glmnet(family="gaussian")``, fits the response
        scaled by `response_scale_`, which divides the ridge part of the
        penalty by it; `RegGLM` (as R's ``family=gaussian()``) does not. The
        estimator's `lambda_val` and `alpha` are those for which `RegGLM`
        solves the same problem as this path at `lambda_val`:
        `lambda_val * alpha` for the lasso part and
        `lambda_val * (1 - alpha) / response_scale_` for the ridge part.

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
        estimator, state = super().get_fixed_lambda(lambda_val)
        lasso = lambda_val * self.alpha
        ridge = lambda_val * (1 - self.alpha) / self.response_scale_
        estimator.lambda_val = lasso + ridge
        estimator.alpha = lasso / (lasso + ridge) if lasso + ridge > 0 else self.alpha
        return estimator, state

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for Gaussian regression.

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

        if offset is None:
            is_offset = False
        else:
            offset = np.asarray(offset).astype(float)
            response = response - offset # makes a copy, does not modify y
            is_offset = True

        # compute nulldeviance

        y = response # shorthand
        ybar = (y * sample_weight).sum() / sample_weight.sum()
        nulldev = ((y - ybar)**2 * sample_weight).sum() / sample_weight.sum()

        if nulldev == 0:
            raise ValueError("response is constant; GaussNet fails at standardization step")

        # as R's glmnet, the C++ path fits y / response_scale_ (its weighted
        # standard deviation, root mean square without an intercept) at
        # lambda / response_scale_: the ridge term is divided by it
        w = sample_weight / sample_weight.sum()
        m2 = (w * y**2).sum()
        self.response_scale_ = np.sqrt(m2 - (w * y).sum()**2 if self.fit_intercept else m2)

        _args = super()._wrapper_args(design,
                                      y.copy(), # it will otherwise be scaled
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # add 'ka' 
        if self.covariance is None:
            nvars = design.X.shape[1]
            if nvars < 500:
                self.covariance = True
            else:
                self.covariance = False

        _args['ka'] = {True:1,
                       False:2}[self.covariance]

        # Gaussian calls it rsq
        _args['rsq'] = _args['dev']
        del(_args['dev'])

        # having problems with a read-only weight possibly
        _args['w'] = _args['w'].copy()
        # doesn't use nulldev
        del(_args['nulldev'])
        return _args
