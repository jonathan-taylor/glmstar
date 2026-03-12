from itertools import product
import logging
import warnings

from typing import Union, Optional
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error)
from .fastnet import MultiFastNetMixin

from .._multigaussnet import multigaussnet as _dense
from .._multigaussnet import spmultigaussnet as _sparse

from ..scoring import Scorer

@dataclass
class MultiClassFamily(object):
    """Family specification for multi-response regression."""

    def _default_scorers(self):
        """Get default scorers for multi-response regression.
        
        Returns
        -------
        list
            List of default scorers.
        """
        return [mse_scorer, mae_scorer]

@dataclass
class MultiGaussNet(MultiFastNetMixin):
    """MultiGaussNet estimator for multi-response Gaussian regression using the FastNet path algorithm.

    This class implements the regularization path for multi-response Gaussian regression
    models using coordinate descent.

    Parameters
    ----------
    response_id : int, str, or list, optional
        Identifiers for response columns.
    offset_id : int, str, or list, optional
        Identifiers for offset columns.
    standardize_response : bool, default=False
        Whether to standardize the response.
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
    coefs_ : ndarray of shape (n_lambda, n_responses, n_features)
        Fitted coefficients across the path.
    intercepts_ : ndarray of shape (n_lambda, n_responses)
        Fitted intercepts across the path.
    lambda_values_ : ndarray of shape (n_lambda,)
        The sequence of lambda values used.
    """

    response_id: Optional[Union[int,str,list]] = None
    offset_id: Optional[Union[int,str,list]] = None
    standardize_response: bool = False
    _family: MultiClassFamily = field(default_factory=MultiClassFamily)
    _dense = _dense
    _sparse = _sparse

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        """Extract fitted coefficients, intercepts, and related statistics for multi-response Gaussian models.

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
        # gaussian fit calls it rsq
        # should be `dev` for sumamry

        self._fit['dev'] = self._fit['rsq']
        return super()._extract_fits(X_shape, response_shape)

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        """Prepare and validate data arrays for multi-response Gaussian regression.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target matrix.
        check : bool, default=True
            Whether to check input validity.

        Returns
        -------
        tuple
            Tuple of (X, y, response, offset, weight).
        """
        X, y, response, offset, weight = super().get_data_arrays(X,
                                                                 y,
                                                                 check=check)
        return X, y, np.asfortranarray(response), offset, weight

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for multi-response Gaussian regression.

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
            # be sure to copy just in case C++ modifies in place
            # (it does sometimes modify offset)
            response = np.asfortranarray(response.copy())            
        else:
            offset = np.asarray(offset).astype(float)
            response = response - offset # make a copy, do not modify 
            is_offset = True

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        _args['jsd'] = int(self.standardize_response)
        _args['rsq'] = _args['dev']

        _args['w'] = _args['w'].copy() # pandas returns a non-writeable ref
        
        del(_args['dev'])
        del(_args['nulldev'])

        return _args

    def _offset_predictions(self,
                            predictions,
                            offset):
        """Add offset to predictions.
        
        Parameters
        ----------
        predictions : array-like
            Predictions array.
        offset : array-like
            Offset array.
            
        Returns
        -------
        array-like
            Predictions with offset added.
        """

        return predictions + offset[:,None,:]

# for CV scores

def _MSE(y, y_hat, sample_weight):
    """Compute mean squared error for multi-response data.
    
    Parameters
    ----------
    y : array-like
        True values.
    y_hat : array-like
        Predicted values.
    sample_weight : array-like
        Sample weights.
        
    Returns
    -------
    float
        Mean squared error.
    """
    return (mean_squared_error(y,
                               y_hat,
                               sample_weight=sample_weight) *
            y_hat.shape[1])

def _MAE(y, y_hat, sample_weight):
    """Compute mean absolute error for multi-response data.
    
    Parameters
    ----------
    y : array-like
        True values.
    y_hat : array-like
        Predicted values.
    sample_weight : array-like
        Sample weights.
        
    Returns
    -------
    float
        Mean absolute error.
    """
    return (mean_absolute_error(y,
                                y_hat,
                                sample_weight=sample_weight) *
            y_hat.shape[1])


mse_scorer = Scorer(name='Mean Squared Error',
                    score=_MSE,
                    maximize=False)
mae_scorer = Scorer(name='Mean Absolute Error',
                    score=_MAE,
                    maximize=False)

