import numpy as np
import pandas as pd

from sklearn.base import clone

from .base import Design
from .glm import (compute_grad,
                  GLM)

def bootstrap_GLM(X, 
                  Df,
                  glm=None,
                  active_set=None, 
                  inactive_set=None,
                  rng=None, B=500):
    """
    Performs a bootstrap procedure for a Generalized Linear Model (GLM).

    This function generates bootstrap samples of coefficients and gradients
    by resampling with replacement from the data.

    Parameters
    ----------
    X : array-like
        The input data matrix.
    Df : pd.DataFrame
        DataFrame containing the response and possibly weights and offsets.
    glm : GLM object, optional
        A GLM object to use for fitting. If None, a default GLM is created.
    active_set : list of int, optional
        Indices of the active set of features.
    inactive_set : list of int, optional
        Indices of the inactive set of features.
    rng : np.random.Generator, optional
        A random number generator.
    B : int, optional
        The number of bootstrap samples to generate.

    Returns
    -------
    coefs : np.ndarray
        An array of bootstrap coefficients.
    grads : np.ndarray
        An array of bootstrap gradients for the inactive set.
    """
    
    if glm is None:
        glm = GLM()

    n = X.shape[0]
    
    if rng is None:
        rng = np.random.default_rng()

    if active_set is not None:
        if type(X) == np.ndarray:
            X_a = X[:,active_set]
        else:
            X_a = X.loc[:,active_set]
    else:
        X_a = X

    # find a warm start
    
    glm_ = clone(glm)
    warm_state = glm_.fit(X_a, Df).state_
    
    _, Y, R, O, W = glm.get_data_arrays(X, Df)

    prob = W / W.sum()
    D = Design(X,
               W,
               standardize=glm.standardize,
               intercept=glm.fit_intercept)

    coefs = []
    grads = []

    for _ in range(B):

        idx = rng.choice(n, n, replace=True, p=prob)
        W_idx = np.bincount(idx, minlength=X.shape[0])

        glm_ = clone(glm)
        glm_.summarize = True
        glm_.weight_id = f'boot_weight_{id(glm_)}'
        glm_.response_id = f'response_{id(glm_)}'
        Df_star = pd.DataFrame({f'response_{id(glm_)}': R,
                                f'boot_weight_{id(glm_)}': W_idx})
        if O is not None:
            Df_star[f'offset_{id(glm_)}'] = O
            glm_.offset_id = f'offset_{id(glm_)}'

        glm_.fit(X_a,
                 Df_star,
                 warm_state=warm_state)
        coefs.append(np.hstack([glm_.intercept_, glm_.coef_.copy()]))

        full_coef = np.zeros(X.shape[1])
        if active_set is not None:
            full_coef[active_set] = glm_.coef_
        else:
            full_coef[:] = glm_.coef_
        if inactive_set is not None:
            D_star = Design(X,
                            W_idx,
                            standardize=glm_.standardize,
                            intercept=glm_.fit_intercept)
            grad = compute_grad(glm_, 
                                glm_.intercept_,
                                full_coef,
                                D_star,
                                R,
                                offset=O,
                                sample_weight=W_idx)[0] 
            # drop the coef and keep inactive coordinates
            grads.append(grad[1:][inactive_set]) 
        else:
            grads.append(np.nan)

    return np.array(coefs), np.array(grads)

def parametric_GLM(X, 
                   Df,
                   glm=None,
                   active_set=None, 
                   inactive_set=None,
                   rng=None, B=500):
    """
    Performs a parametric bootstrap for a Generalized Linear Model (GLM).

    This function generates bootstrap samples by simulating new responses from
    the fitted model.

    Parameters
    ----------
    X : array-like
        The input data matrix.
    Df : pd.DataFrame
        DataFrame containing the response and possibly weights and offsets.
    glm : GLM object, optional
        A GLM object to use for fitting. If None, a default GLM is created.
    active_set : list of int, optional
        Indices of the active set of features.
    inactive_set : list of int, optional
        Indices of the inactive set of features.
    rng : np.random.Generator, optional
        A random number generator.
    B : int, optional
        The number of bootstrap samples to generate.

    Returns
    -------
    coefs : np.ndarray
        An array of bootstrap coefficients.
    grads : np.ndarray
        An array of bootstrap gradients for the inactive set.
    """
    
    if glm is None:
        glm = GLM()

    n = X.shape[0]
    
    if rng is None:
        rng = np.random.default_rng()

    if active_set is not None:
        if type(X) == np.ndarray:
            X_a = X[:,active_set]
        else:
            X_a = X.loc[:,active_set]
    else:
        X_a = X

    # find a warm start
    
    glm_ = clone(glm)
    warm_state = glm_.fit(X_a, Df, dispersion=None).state_
    
    _, Y, R, O, W = glm.get_data_arrays(X, Df)

    D = Design(X,
               W,
               standardize=glm.standardize,
               intercept=glm.fit_intercept)

    coefs = []
    grads = []

    lin_pred = glm_.predict(X, prediction_type='link')
    fam = glm_._family

    for _ in range(B):

        Y_star = fam.rvs(lin_pred,
                         scale=np.sqrt(glm_.dispersion_))

        glm_ = clone(glm)
        glm_.summarize = True
        glm_.weight_id = f'boot_weight_{id(glm_)}'
        glm_.response_id = f'response_{id(glm_)}'
        Df_star = pd.DataFrame({f'response_{id(glm_)}': Y_star,
                                f'boot_weight_{id(glm_)}': W})
        if O is not None:
            Df_star[f'offset_{id(glm_)}'] = O
            glm_.offset_id = f'offset_{id(glm_)}'

        glm_.fit(X_a,
                 Df_star,
                 warm_state=warm_state)
        coefs.append(np.hstack([glm_.intercept_, glm_.coef_.copy()]))

        full_coef = np.zeros(X.shape[1])
        if active_set is not None:
            full_coef[active_set] = glm_.coef_
        else:
            full_coef[:] = glm_.coef_
        if inactive_set is not None:
            grad = compute_grad(glm_, 
                                glm_.intercept_,
                                full_coef,
                                D,
                                Y_star,
                                offset=O,
                                sample_weight=W)[0] 
            # drop the coef and keep inactive coordinates
            grads.append(grad[1:][inactive_set]) 
        else:
            grads.append(np.nan)

    return np.array(coefs), np.array(grads)
