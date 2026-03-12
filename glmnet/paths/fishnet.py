import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from statsmodels.genmod.families import family as sm_family

from .fastnet import FastNetMixin
from ..glm import GLMFamilySpec

from .._fishnet import fishnet as _dense
from .._fishnet import spfishnet as _sparse

"""
Implements the FishNet path algorithm for Poisson regression models.
Provides the FishNet estimator class using the FastNetMixin base.
"""

@dataclass
class FishNet(FastNetMixin):
    """FishNet estimator for Poisson regression using the FastNet path algorithm."""

    _dense = _dense
    _sparse = _sparse

    # private methods

    def __post_init__(self):
        """Initialize the FishNet estimator and set the GLM family to Poisson."""
        self._family = GLMFamilySpec(base=sm_family.Poisson())

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        """Prepare and validate data arrays for Poisson regression.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        check : bool, default=True
            Whether to check input validity.

        Returns
        -------
        tuple
            Tuple of (X, y, response, offset, weight).
        """
        X, y, response, offset, weight = super().get_data_arrays(X, y, check=check)
        if np.any(response < 0):
            raise ValueError("negative responses encountered;  not permitted for Poisson family")
        response = np.asarray(response, float).copy()
        return X, y, response, offset, weight
    
    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for Poisson regression.

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
            offset = 0. * response

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        _args['g'] = np.asfortranarray(offset.reshape((-1,1)).copy())

        _args['w'] = _args['w'].copy()
        
        return _args

