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

    This class implements the regularization path for binomial (logistic) regression
    models using coordinate descent.

    Parameters
    ----------
    modified_newton : bool, default=False
        Whether to use the modified Newton method.
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
    classes_ : ndarray
        The classes labels.
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

        For binomial regression, the response can be specified as a 1D array of labels,
        or as a 2D array of shape (n_samples, 2) containing pairs of (trials, successes).
        If provided as (trials, successes), `sample_weight` will be multiplied by the 
        number of trials, and the response will be transformed into proportions. This
        assumption is documented because users might mistakenly use trials as weights
        without accounting for them in the data shape.

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
        
        if response.ndim == 2 and response.shape[1] == 2:
            trials = response[:, 0]
            successes = response[:, 1]
            if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
                raise ValueError("For (trials, successes) input, trials must be > 0 and 0 <= successes <= trials.")
            
            weight = weight * trials
            labels = successes / trials
            self.classes_ = self._family.classes_ = np.array([0, 1])
        else:
            if response.ndim == 2 and response.shape[1] == 1:
                response = response.ravel()
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
        if np.issubdtype(_args['y'].dtype, np.floating) or len(np.unique(_args['y'])) > 2:
            p = _args['y'].ravel()
            y_onehot = np.column_stack([1 - p, p])
        else:
            encoder = OneHotEncoder(sparse_output=False)
            y_onehot = encoder.fit_transform(_args['y'].reshape(-1, 1))
            
        _args['y'] = np.asfortranarray(y_onehot)

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
