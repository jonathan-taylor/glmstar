from dataclasses import (dataclass,
                         asdict,
                         field)
import numpy as np
from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .base import _get_design, DiagonalOperator
from .scoring import (Scorer,
                      mae_scorer,
                      mse_scorer,
                      accuracy_scorer,
                      auc_scorer,
                      aucpr_scorer,
                      ungrouped_mse_scorer,
                      ungrouped_mae_scorer)

@dataclass
class GLMFamilySpec(object):
    """Specification for GLM family and link function.
    
    Parameters
    ----------
    base : sm_family.Family, default=sm_family.Gaussian
        The base family from statsmodels.
    """
    
    base: sm_family.Family = field(default_factory=sm_family.Gaussian)

    def __post_init__(self):

        self.is_gaussian = False
        if (isinstance(self.base, sm_family.Gaussian) and
            isinstance(self.base.link, sm_links.Identity)):
            self.is_gaussian = True

        self.is_binomial = False
        if isinstance(self.base, sm_family.Binomial):
            self.is_binomial = True

    @staticmethod
    def from_family(family,
                    response):
        """Create GLMFamilySpec from family object.
        
        Parameters
        ----------
        family : sm_family.Family or GLMFamilySpec
            Family object or specification.
        response : array-like
            Response variable (unused, kept for compatibility).
            
        Returns
        -------
        GLMFamilySpec
            Family specification object.
        """
        if isinstance(family, sm_family.Family):
            return GLMFamilySpec(base=family)
        elif isinstance(family, GLMFamilySpec):
            return family


    def link(self,
             mean_parameter):
        """Apply link function to mean parameter.
        
        Parameters
        ----------
        mean_parameter : array-like
            Mean parameter values.
            
        Returns
        -------
        array-like
            Linear predictor values.
        """
        mu = mean_parameter # shorthand
        return self.base.link(mu)
    
    def deviance(self,
                 response,
                 mean_parameter,
                 sample_weight=None):
        """Compute deviance.
        
        Parameters
        ----------
        response : array-like
            Response variable.
        mean_parameter : array-like
            Mean parameter values.
        sample_weight : array-like, optional
            Sample weights.
            
        Returns
        -------
        float
            Deviance value.
        """
        if sample_weight is not None:
            return self.base.deviance(response,
                                      mean_parameter,
                                      freq_weights=sample_weight)
        else:
            return self.base.deviance(response,
                                      mean_parameter)

    def null_fit(self,
                 response,
                 fit_intercept=True,
                 sample_weight=None,
                 offset=None):
        """Fit null model (intercept only).
        
        Parameters
        ----------
        response : array-like
            Response variable.
        fit_intercept : bool, default=True
            Whether to fit intercept.
        sample_weight : array-like, optional
            Sample weights.
        offset : array-like, optional
            Offset values.
            
        Returns
        -------
        GLMState
            Fitted null state of GLM.
        """

        sample_weight = np.asarray(sample_weight)
        y = np.asarray(response)

        if offset is None:
            offset = np.zeros(y.shape[0])
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        if fit_intercept:

            # solve a one parameter problem

            X1 = np.ones((y.shape[0], 1))
            D = _get_design(X1,
                            sample_weight,
                            standardize=False,
                            intercept=False)
            
            state = GLMState(np.zeros(1),
                             0)
            state.update(D,
                         self,
                         offset,
                         None)

            for i in range(10):

                z, w = self.get_response_and_weights(state,
                                                     y,
                                                     offset,
                                                     sample_weight)
                newcoef = (z*w).sum() / w.sum()
                state = GLMState(np.array([newcoef]),
                                 0)
                state.update(D,
                             self,
                             offset,
                             None)
        else:
            state = GLMState(np.zeros(1), 0)
            state.link_parameter = offset
            state.mean_parameter = self.base.link.inverse(state.link_parameter)
        return state

    def get_null_deviance(self,
                          response,
                          sample_weight=None,
                          offset=None,
                          fit_intercept=True):
        """Get null deviance and state.
        
        Parameters
        ----------
        response : array-like
            Response variable.
        sample_weight : array-like, optional
            Sample weights.
        offset : array-like, optional
            Offset values.
        fit_intercept : bool, default=True
            Whether to fit intercept.
            
        Returns
        -------
        null_state : GLMState
            Fitted null state of GLM.
        deviance : float
            Null deviance value.
        """
        state0 = self.null_fit(response,
                               fit_intercept=fit_intercept,
                               sample_weight=sample_weight,
                               offset=offset)
        D = self.deviance(response,
                          state0.mean_parameter,
                          sample_weight=sample_weight)
        return state0, D

    def get_response_and_weights(self,
                                 state,
                                 response,
                                 offset,
                                 sample_weight):
        """Get pseudo-response and weights for Newton step.
        
        Parameters
        ----------
        state : GLMState
            State of GLM.
        response : array-like
            Response variable.
        offset : array-like
            Offset values.
        sample_weight : array-like
            Sample weights.
            
        Returns
        -------
        pseudo_response : np.ndarray
            Pseudo-response for (quasi) Newton step.
        newton_weights : np.ndarray
            Weights to be used for diagonal in Newton step.
        """

        y = response # shorthand
        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.link_parameter)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        newton_weights = sample_weight * dmu_deta**2 / varmu

        pseudo_response = state.eta + (y - state.mu) / dmu_deta

        return pseudo_response, newton_weights

    def information(self,
                    state,
                    sample_weight=None):
        """Compute information matrix.
        
        Parameters
        ----------
        state : GLMState
            State of GLM.
        sample_weight : array-like, optional
            Sample weights.
            
        Returns
        -------
        information : DiagonalOperator
            Diagonal information matrix of the response vector for
            `state.mean_pararmeter`.
        """

        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.link_parameter)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        W = dmu_deta**2 / varmu
        if sample_weight is not None:
            W *= sample_weight
            
        n = W.shape[0]
        W = W.reshape(-1)
        return DiagonalOperator(W)

    # Private methods

    def _default_scorers(self):
        """Construct default scorers for GLM.
        
        Returns
        -------
        list
            List of default scorers.
        """

        fam_name = self.base.__class__.__name__

        def _dev(y, yhat, sample_weight):
            return self.deviance(y, yhat, sample_weight) / y.shape[0]
        dev_scorer = Scorer(name=f'{fam_name} Deviance',
                            score=_dev,
                            maximize=False)
        
        scorers_ = [dev_scorer,
                    mse_scorer,
                    mae_scorer,
                    ungrouped_mse_scorer,
                    ungrouped_mae_scorer]

        if isinstance(self.base, sm_family.Binomial):
            scorers_.extend([accuracy_scorer,
                             auc_scorer,
                             aucpr_scorer])

        return scorers_

    def _get_null_state(self,
                        null_fit,
                        nvars):
        """Get null state for given number of variables.
        
        Parameters
        ----------
        null_fit : GLMState
            Null fit state.
        nvars : int
            Number of variables.
            
        Returns
        -------
        GLMState
            Null state with correct dimensions.
        """
        coefold = np.zeros(nvars)   # initial coefs = 0
        state = GLMState(coef=coefold,
                         intercept=null_fit.intercept)
        state.mean_parameter = null_fit.mean_parameter
        state.link_parameter = null_fit.link_parameter
        return state

    def predict(self,
                linpred,
                prediction_type='response'):
        """Make predictions.
        
        Parameters
        ----------
        linpred : array-like
            Linear predictor values.
        prediction_type : str, default='response'
            Type of prediction ('response', 'link' or 'class').
            
        Returns
        -------
        array-like
            Predictions.
        """

        if prediction_type == 'link':
            return linpred
        elif prediction_type == 'response':
            return self.base.link.inverse(linpred)
        else:
            raise ValueError("prediction should be one of 'response' or 'link'")


@dataclass
class BinomFamilySpec(GLMFamilySpec):
    """Binomial family specification.
    
    Parameters
    ----------
    base : sm_family.Family, default=sm_family.Binomial
        The base family from statsmodels.
    """

    base: sm_family.Family = field(default_factory=sm_family.Binomial)

    def predict(self,
                linpred,
                prediction_type='response'):
        """Make predictions for binomial family.
        
        Parameters
        ----------
        linpred : array-like
            Linear predictor values.
        prediction_type : str, default='response'
            Type of prediction ('response', 'link', or 'class').
            
        Returns
        -------
        array-like
            Predictions.
        """

        if prediction_type == 'link':
            return linpred
        elif prediction_type == 'response':
            return self.base.link.inverse(linpred)
        elif prediction_type == 'class':
            pi_hat = self.base.link.inverse(linpred)
            _classes = (pi_hat > 0.5).astype(int)
            if hasattr(self, "classes_"):
                _classes = self.classes_[_classes]
            return _classes
        else:
            raise ValueError("prediction should be one of 'response', 'link' or 'class'")


@dataclass
class GLMState(object):
    """State of GLM model.
    
    Parameters
    ----------
    coef : np.ndarray
        Coefficient vector.
    intercept : np.ndarray
        Intercept value.
    obj_val : float, default=np.inf
        Objective function value.
    pmin : float, default=1e-9
        Minimum probability for binomial family.
    """

    coef: np.ndarray
    intercept: np.ndarray
    obj_val: float = np.inf
    pmin: float = 1e-9
    
    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset,
               objective=None):
        """Update state with new parameters.
        
        Parameters
        ----------
        design : Design
            Design matrix.
        family : GLMFamilySpec
            Family specification.
        offset : array-like, optional
            Offset values.
        objective : callable, optional
            Objective function to evaluate.
        """

        family = family.base
        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = family.link.inverse(self.link_parameter)

        # shorthand
        self.mu = self.mean_parameter 
        self.eta = self.linear_predictor 

        if isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):
        """Compute log-likelihood score.
        
        Parameters
        ----------
        family : GLMFamilySpec
            Family specification.
        y : array-like
            Response variable.
        sample_weight : array-like
            Sample weights.
            
        Returns
        -------
        array-like
            Score values.
        """

        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.link_parameter)
        
        # compute working residual
        r = (y - self.mu) 
        return sample_weight * r * dmu_deta / varmu

