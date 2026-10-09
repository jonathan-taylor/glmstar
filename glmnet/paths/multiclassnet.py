from itertools import product
import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y
from sklearn.metrics import (accuracy_score,
                             zero_one_loss)

from .fastnet import MultiFastNetMixin
from .._utils import _jerr_lognet

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

from ..scoring import (Scorer,
                       multinomial_deviance_scorer)

@dataclass
class MultiClassFamily(object):
    """Family specification for multinomial classification."""

    def _default_scorers(self):
        """Get default scorers for multinomial classification.
        
        Returns
        -------
        list
            List of default scorers.
        """
        return [accuracy_scorer,
                misclass_scorer,
                deviance_scorer,
                mse_scorer,
                mae_scorer]

@dataclass
class MultiClassNet(MultiFastNetMixin):
    """MultiClassNet estimator for multinomial classification using the FastNet path algorithm.

    This class implements the regularization path for multinomial classification
    models using coordinate descent.

    Parameters
    ----------
    standardize_response : bool, default=False
        Whether to standardize the response.
    grouped : bool, default=False
        Whether to use grouped multinomial loss.
    univariate_beta : bool, default=True
        Whether to use univariate beta updates.
    type_logistic : {'Newton', 'modified_Newton'}, default='Newton'
        Type of logistic solver.
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
    coefs_ : ndarray of shape (n_lambda, n_classes, n_features)
        Fitted coefficients across the path.
    intercepts_ : ndarray of shape (n_lambda, n_classes)
        Fitted intercepts across the path.
    lambda_values_ : ndarray of shape (n_lambda,)
        The sequence of lambda values used.
    categories_ : ndarray
        The classes labels.
    """

    standardize_response: bool = False
    grouped: bool = False
    univariate_beta: bool = True
    type_logistic: Literal['Newton', 'modified_Newton'] = 'Newton'
    _family: MultiClassFamily = field(default_factory=MultiClassFamily)
    _dense = _dense
    _sparse = _sparse
    _jerr_message = staticmethod(_jerr_lognet)

    def predict(self,
                X,
                prediction_type='response', # ignored except checking valid
                interpolation_grid=None,
                offset=None
                ):
        """Predict class probabilities or logits for multinomial classification.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        prediction_type : str, default='response'
            Type of prediction ('response' for probabilities, 'link' for logits).
        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.
        offset : array-like, optional
            Offset for the rows of `X`, of shape `(n_samples, n_classes)`,
            added to the logits (R's `newoffset`). If omitted, no offset is used.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        value = super().predict(X,
                                interpolation_grid=interpolation_grid,
                                prediction_type='link',
                                offset=offset)
        if prediction_type == 'response':
            # value is (n, nlambda, K), or (n, K) for a scalar interpolation_grid
            value = value - value.max(-1, keepdims=True)
            exp_value = np.exp(value)
            value = exp_value / exp_value.sum(-1, keepdims=True)
        elif prediction_type == 'class':
            int_class = np.argmax(value, -1)
            value = self.categories_[int_class]
        return value
        
    # private methods

    def _offset_predictions(self,
                            predictions,
                            offset):
        """Apply offset to predictions for multinomial classification.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values before offset.
        offset : np.ndarray
            Offset values.

        Returns
        -------
        np.ndarray
            Offset-adjusted predictions.
        """
        value = np.log(predictions) + offset[:,None,:]
        _max = value.max(-1)
        value = value - _max[:,:,None]
        exp_value = np.exp(value)
        value = exp_value / exp_value.sum(-1)[:,:,None]
        return value

    def get_data_arrays(self, X, y, check=True):
        """Prepare and validate data arrays for multinomial classification.

        For multinomial classification, the response can be specified as a 1D array of 
        class labels, or as a 2D array of counts where each column represents a class 
        and each row is an observation. If provided as counts, `sample_weight` will be 
        multiplied by the number of trials (row sums), and the response will be 
        transformed into proportions. This assumption is documented because users might 
        mistakenly use the trials as `sample_weight` without accounting for them in the data shape.

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
            Tuple of (X, y, y_onehot, offset, weight).
        """

        X, y, response, offset, weight = super().get_data_arrays(X, y, check=check)
        
        if response.ndim == 2 and response.shape[1] > 1:
            trials = response.sum(axis=1)
            if np.any(response < 0) or np.any(trials <= 0):
                raise ValueError("Response matrix must contain non-negative counts and row sums must be > 0.")
            weight = weight * trials
            y_onehot = np.asfortranarray(response / trials[:, None])
            self.categories_ = np.arange(response.shape[1])
        else:
            if response.ndim == 2 and response.shape[1] == 1:
                response = response.ravel()
            encoder = OneHotEncoder(sparse_output=False)
            y_onehot = np.asfortranarray(encoder.fit_transform(response.reshape((-1,1))))
            self.categories_ = encoder.categories_[0]

        return X, y, y_onehot, offset, weight

    def predict_proba(self,
                      X,
                      interpolation_grid=None,
                      offset=None):
        """
        Probability estimates for a LogNet model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.

        offset : array-like, optional
            Offset for the rows of `X` (see `predict`).

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        return self.predict(X,
                            interpolation_grid=interpolation_grid,
                            prediction_type='response',
                            offset=offset)

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        """Extract fitted coefficients, intercepts, and related statistics for multinomial models.

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
        # center the intercepts -- any constant
        # added does not affect class probabilities
        
        self._fit['a0'] = self._fit['a0'] - self._fit['a0'].mean(0)[None,:]
        return super()._extract_fits(X_shape, response_shape)

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for multinomial classification.

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

        if offset is None:
            offset = response * 0.
        if offset.shape != response.shape:
            raise ValueError('offset shape should match one-hot response shape')
        offset = np.asfortranarray(offset.copy())

        # add 'kopt' 
        _args['kopt'] = {'Newton':0,
                         'modified_Newton':1}[self.type_logistic]
        # if grouped, we set kopt to 2
        if self.grouped:
            _args['kopt'] = 2

        # add 'g'
        _args['g'] = offset

        # take care of weights
        _args['y'] = np.asfortranarray(_args['y'] * sample_weight[:,None])

        # remove w
        del(_args['w'])

        return _args

# for CV scores

def _misclass(y, p_hat, sample_weight):
    """Compute misclassification error for multinomial classification.
    
    Parameters
    ----------
    y : array-like
        True one-hot encoded labels.
    p_hat : array-like
        Predicted probabilities.
    sample_weight : array-like
        Sample weights.
        
    Returns
    -------
    float
        Misclassification error.
    """
    return zero_one_loss(np.argmax(y, -1),
                         np.argmax(p_hat, -1),
                         sample_weight=sample_weight,
                         normalize=True)

def _accuracy_score(y, p_hat, sample_weight):
    """Compute accuracy for multinomial classification.
    
    Parameters
    ----------
    y : array-like
        True one-hot encoded labels.
    p_hat : array-like
        Predicted probabilities.
    sample_weight : array-like
        Sample weights.
        
    Returns
    -------
    float
        Accuracy score.
    """
    return accuracy_score(np.argmax(y, -1),
                          np.argmax(p_hat, -1),
                          sample_weight=sample_weight,
                          normalize=True)

misclass_scorer = Scorer(name='Misclassification Error',
                         score=_misclass,
                         maximize=False)

accuracy_scorer = Scorer(name='Accuracy',
                         score=_accuracy_score,
                         maximize=True)

# clamps predicted probabilities as in R's cv.glmnet
deviance_scorer = multinomial_deviance_scorer()

def _mse(y, p_hat, sample_weight):
    """Squared error summed over classes, as R's cv.glmnet(type.measure="mse")."""
    return np.average(((y - p_hat)**2).sum(-1), weights=sample_weight)

def _mae(y, p_hat, sample_weight):
    """Absolute error summed over classes, as R's cv.glmnet(type.measure="mae")."""
    return np.average(np.fabs(y - p_hat).sum(-1), weights=sample_weight)

mse_scorer = Scorer(name='Mean Squared Error',
                    score=_mse,
                    maximize=False)

mae_scorer = Scorer(name='Mean Absolute Error',
                    score=_mae,
                    maximize=False)
