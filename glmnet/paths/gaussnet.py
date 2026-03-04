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

    Parameters
    ----------
    covariance : bool or None, optional
        Whether to use the covariance update (default: auto based on nvars).
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
