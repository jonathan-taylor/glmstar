import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import (OneHotEncoder,
                                   LabelEncoder)
from statsmodels.genmod.families import family as sm_family

from .fastnet import FastNetMixin
from ..glm import BinomFamilySpec

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

"""
Implements the LogNet path algorithm for binomial (logistic) regression models.
Provides the LogNet estimator class using the FastNetMixin base.
"""

@dataclass
class LogNet(FastNetMixin):
    """LogNet estimator for binomial (logistic) regression using the FastNet path algorithm.

    Parameters
    ----------
    modified_newton : bool, default=False
        Whether to use the modified Newton method.
    """

    modified_newton: bool = False
    _dense = _dense
    _sparse = _sparse

    def __post_init__(self):
        """Initialize the LogNet estimator and set the GLM family to Binomial."""
        self._family = BinomFamilySpec(base=sm_family.Binomial())

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape): # getcoef.R
        """Extract fitted coefficients, intercepts, and related statistics for binary models.

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
        # intercepts will be shape (1,nfits),
        # reshape to (nfits,)
        # specific to binary
        self._fit['a0'] = self._fit['a0'].reshape(-1)
        V = super()._extract_fits(X_shape, response_shape)
        return V

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        """Prepare and validate data arrays for binomial regression.

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
            Tuple of (X, y, labels, offset, weight).
        """
        X, y, response, offset, weight = super().get_data_arrays(X, y, check=check)
        encoder = LabelEncoder()
        labels = np.asfortranarray(encoder.fit_transform(response))
        self.classes_ = self._family.classes_ = encoder.classes_
        if len(encoder.classes_) > 2:
            raise ValueError("BinomialGLM expecting a binary classification problem.")
        return X, y, labels, offset, weight

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for binomial regression.

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

        # adjust dim of offset -- seems necessary to get 1d?

        if offset is None:
            offset = 0. * response # np.zeros(response.shape + (2,))
            
        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L57
        offset = np.column_stack([offset,-offset])
        offset = np.asfortranarray(offset)

        n_samples, n_features = design.X.shape

        # add 'kopt' 
        _args['kopt'] = int(self.modified_newton)

        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L42
        nc = 1

        # add 'g'
        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L65
        _args['g'] = offset[:,0]

        # fix intercept and coefs

        _args['a0'] = np.asfortranarray(np.zeros((nc, self.nlambda), float))
        _args['ca'] = np.zeros((n_features*self.nlambda*nc, 1))

        # reshape y
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(_args['y']))
        _args['y'] = y_onehot

        _args['y'] *= sample_weight[:,None]
        # from https://github.com/trevorhastie/glmnet/blob/master/R/lognet.R#L43
        _args['y'] = np.asfortranarray(_args['y'][:,::-1])

        # remove w
        del(_args['w'])
        
        return _args

    def predict_proba(self,
                      X,
                      interpolation_grid=None):
        """
        Probability estimates for a LogNet model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        prob_1 = self.predict(X,
                              interpolation_grid=interpolation_grid,
                              prediction_type='response')
        result = np.empty(prob_1.shape + (2,))
        result[:,:,1] = prob_1
        result[:,:,0] = 1 - prob_1

        return result
