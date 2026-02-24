from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
import warnings
import logging
   
import numpy as np

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .base import _get_design, Penalty

from .elnet import (ElNet,
                    ElNetControl,
                    ElNetSpec)
from .glm import GLM
from .family import (GLMFamilySpec,
                     BinomFamilySpec)


@dataclass
class RegGLMControl(ElNetControl):
    """
    Control parameters for regularized GLM fitting.
    
    Parameters
    ----------
    thresh: float
        Convergence threshold for coordinate descent. Each inner
        coordinate-descent loop continues until the maximum change in the
        objective after any coefficient update is less than thresh times
        the null deviance. Default value is `1e-10`.
    mxitnr: int
        Maximum number of quasi Newton iterations.
    epsnr: float
        Tolerance for quasi Newton iterations.
    big: float
        A large float, effectively `np.inf`.
    logging: bool
        Write info and debug messages to log?
    """
    thresh: float = 1e-10
    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35
    logging: bool = False


@dataclass
class RegGLMSpec(ElNetSpec):
    """
    Specification for regularized GLM models.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    control: RegGLMControl
        Parameters to control the solver.
    """
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    control: RegGLMControl = field(default_factory=RegGLMControl)


@dataclass
class RegGLMResult(object):
    """
    Result object for regularized GLM fitting.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    offset: bool
        Whether offset was used in fitting.
    converged: bool
        Did the algorithm converge?
    boundary: bool
        Was backtracking required due to getting near boundary of valid mean / natural parameters.
    obj_function: float
        Value of objective function (deviance + penalty).
    """
    family: GLMFamilySpec
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float


@dataclass
class ElNetRegularizer(Penalty):
    """
    Elastic Net regularizer for GLM fitting.
    
    Parameters
    ----------
    fit_intercept: bool
        Should intercept be fitted (default=True) or set to zero (False)?
    warm_state: dict
        Warm start state.
    nvars: int
        Number of variables.
    control: RegGLMControl
        Control parameters.
    exclude: list
        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    """
    fit_intercept: bool = False
    warm_state: dict = field(default_factory=dict)
    nvars: InitVar[Optional[int]] = None
    control: InitVar[Optional[RegGLMControl]] = None
    exclude: list = field(default_factory=list)

    def __post_init__(self, nvars, control):
        """
        Initialize the regularizer after creation.
        
        Parameters
        ----------
        nvars: int
            Number of variables.
        control: RegGLMControl
            Control parameters.
        """

        if self.penalty_factor is None:
            penalty_factor_ = np.ones(nvars)
        else:
            penalty_factor_ = self.penalty_factor

        penalty_factor_ *= nvars / penalty_factor_.sum() 
        self.penalty_factor_ = penalty_factor_
        
        self.elnet_estimator = ElNet(lambda_val=self.lambda_val,
                                     alpha=self.alpha,
                                     control=control,
                                     lower_limits=self.lower_limits,
                                     upper_limits=self.upper_limits,
                                     fit_intercept=self.fit_intercept,
                                     penalty_factor=self.penalty_factor,
                                     standardize=False,
                                     exclude=self.exclude)

    def half_step(self,
                  state,
                  oldstate):
        """
        Compute half step between two states.
        
        Parameters
        ----------
        state: GLMState
            Current state.
        oldstate: GLMState
            Previous state.
            
        Returns
        -------
        GLMState
            Half step state.
        """
        klass = oldstate.__class__
        return klass(0.5 * (oldstate.coef + state.coef),
                     0.5 * (oldstate.intercept + state.intercept))

    def _debug_msg(self,
                   state):
        """Return debug message for state."""
        return f'Coef: {state.coef}, Intercept: {state.intercept}, Objective: {state.obj_val}'

    def check_state(self,
                    state):
        """
        Check state for validity.
        
        Parameters
        ----------
        state: GLMState
            State to check.
            
        Raises
        ------
        ValueError
            If state contains NaN values.
        """
        if np.any(np.isnan(state.coef)):
            raise ValueError('coef has NaNs')
        if np.isnan(state.intercept):
            raise ValueError('intercept is NaN')

    def newton_step(self,
                    design,
                    pseudo_response,
                    normed_sample_weight,
                    cur_state):
        """
        Perform Newton step for regularized GLM fitting.
        
        Parameters
        ----------
        design: Design
            Design matrix.
        pseudo_response: np.ndarray
            Pseudo response for IRLS.
        normed_sample_weight: np.ndarray
            Normalized sample weights.
        cur_state: GLMState
            Current state.
            
        Returns
        -------
        GLMState
            Updated state.
        """
        z = pseudo_response
        # make sure to set lambda_val to self.lambda_val
        self.elnet_estimator.lambda_val = self.lambda_val
        
        warm = (cur_state.coef,
                cur_state.intercept,
                cur_state.linear_predictor) # just X\beta -- doesn't include offset

        elnet_fit = self.elnet_estimator.fit(design,
                                             z,
                                             sample_weight=normed_sample_weight,
                                             warm=warm,
                                             check=False)
        
        klass = cur_state.__class__
        self.warm_state = klass(elnet_fit.raw_coef_,
                                elnet_fit.raw_intercept_)

        return self.warm_state

    def objective(self, state):
        """
        Compute objective value (elastic net penalty).
        
        Parameters
        ----------
        state: GLMState
            Current state.
            
        Returns
        -------
        float
            Objective value (elastic net penalty).
        """
        lasso = self.alpha * np.fabs(state.coef).sum()
        ridge = (1 - self.alpha) * (state.coef**2).sum() / 2
        return self.lambda_val * (lasso + ridge)


@dataclass
class RegGLM(GLM,
             RegGLMSpec):
    """
    Regularized Generalized Linear Model.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    control: RegGLMControl
        Parameters to control the solver.
    """
    control: RegGLMControl = field(default_factory=RegGLMControl)

    def get_LM(self):
        """
        Get the corresponding linear model (GLM without regularization).
        
        Returns
        -------
        GLM
            Linear model instance.
        """
        return GLM(family=self.family,
                   fit_intercept=self.fit_intercept,
                   offset_id=self.offset_id,
                   weight_id=self.weight_id,
                   response_id=self.response_id)

    def _get_regularizer(self,
                         nvars=None):
        """
        Get regularizer for fitting.
        
        Parameters
        ----------
        nvars: int, optional
            Number of variables.
            
        Returns
        -------
        ElNetRegularizer
            Regularizer instance.
        """
        # self.design_ will have been set by now

        if nvars is None:
            check_is_fitted(self, ["design_"])
            nvars = self.design_.X.shape[1]

        return ElNetRegularizer(lambda_val=self.lambda_val,
                                alpha=self.alpha,
                                penalty_factor=self.penalty_factor,
                                lower_limits=self.lower_limits * self.design_.scaling_,
                                upper_limits=self.upper_limits * self.design_.scaling_,
                                fit_intercept=self.fit_intercept,
                                nvars=nvars,
                                control=self.control,
                                exclude=self.exclude)

    def _get_design(self,
                    X,
                    sample_weight):
        """
        Get design matrix.
        
        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        sample_weight: Optional[np.ndarray]
            Sample weights.
            
        Returns
        -------
        Design
            Design matrix.
        """
        return _get_design(X,
                           sample_weight,
                           standardize=self.standardize,
                           intercept=self.fit_intercept)

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 3 options non sklearn API
            warm_state=None,
            check=True,
            fit_null=True):
        """
        Fit a regularized GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: ElNetRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer.
        warm_state: GLMState, optional
            Warm start state.
        check: bool
            Run the `_check` method to validate `(X,y)`.
        fit_null: bool
            Whether to fit null model.

        Returns
        -------
        self: object
            RegGLM class instance.
        """
        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=1,
                    check=check)

        return self


@dataclass
class GaussianRegGLM(RegressorMixin, RegGLM):
    """
    Gaussian regularized GLM for regression.
    
    Parameters
    ----------
    family: GLMFamilySpec
        Specification of one-parameter exponential family, includes some
        additional methods.
    control: RegGLMControl
        Parameters to control the solver.
    """
    def __post_init__(self):
        """
        Post-initialization checks for Gaussian family.
        """
        if not self._family.is_gaussian:
            msg = f'{self.__class__.__name__} expects a Gaussian family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)


@dataclass
class BinomialRegGLM(ClassifierMixin, RegGLM):
    """
    Binomial regularized GLM for classification.
    
    Parameters
    ----------
    family: BinomFamilySpec
        Specification of binomial family.
    control: RegGLMControl
        Parameters to control the solver.
    """
    family: BinomFamilySpec = field(default_factory=BinomFamilySpec)

    def __post_init__(self):
        """
        Post-initialization checks for binomial family.
        """
        if not self._family.is_binomial:
            msg = f'{self.__class__.__name__} expects a Binomial family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):
        """
        Fit a Binomial regularized GLM.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        y: np.ndarray
            Response variable.
        sample_weight: Optional[np.ndarray]
            Sample weights.
        regularizer: ElNetRegularizer, optional
            Regularizer used in fitting the model. Allows for inspection of parameters of regularizer.
        warm_state: GLMState, optional
            Warm start state.
        dispersion: float
            Dispersion parameter of GLM. If family is Gaussian, will be estimated as 
            minimized deviance divided by degrees of freedom.
        check: bool
            Run the `_check` method to validate `(X,y)`.

        Returns
        -------
        self: object
            BinomialRegGLM class instance.
        """
        label_encoder = LabelEncoder().fit(y)
        if len(label_encoder.classes_) > 2:
            raise ValueError("BinomialRegGLM expecting a binary classification problem.")
        self.classes_ = label_encoder.classes_

        y_binary = label_encoder.transform(y)

        return super().fit(X,
                           y_binary,
                           sample_weight=sample_weight,
                           regularizer=regularizer,             # last 4 options non sklearn API
                           dispersion=dispersion,
                           check=check)

    def predict(self, X, prediction_type='class'):
        """
        Predict outcome of corresponding family.

        Parameters
        ----------
        X: Union[np.ndarray, scipy.sparse, DesignSpec]
            Input matrix, of shape `(n_samples, n_features)`; each row is an observation
            vector. If it is a sparse matrix, it is assumed to be
            unstandardized.  If it is not a sparse matrix, a copy is made and
            standardized.
        prediction_type: str
            One of "response", "link" or "class". If "response" return a prediction on the mean scale,
            "link" on the link scale, and "class" as a class label. Defaults to "class".

        Returns
        -------
        prediction: np.ndarray
            Predictions on the mean scale for family of a GLM.
        """
        if not hasattr(self.family, 'base'):
            raise ValueError(f'{self.__class__} expects to have a base family')
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        pred = self._family.predict(linpred,
                                    prediction_type=prediction_type)
        if prediction_type == 'class':
            pred = self._classes[pred]
        return pred

    def predict_proba(self, X):
        """
        Probability estimates for a BinomialRegGLM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
 
        """
        prob_1 = self.predict(X, prediction_type='response')
        result = np.empty((prob_1.shape[0], 2))
        result[:,1] = prob_1
        result[:,0] = 1 - prob_1

        return result
