import numpy as np
import pytest

from glmnet import (GLMNet,
                    compute_grad,
                    GLM)
from glmnet.base import (ScaleOperator,
                         UnscaleOperator)

def test_compute_grad(standardize,
                      fit_intercept,
                      glmnet,
                      scaled_input,
                      scaled_output):

    n, p, prop = 10, 5, 0.8
    scale = 2
    rng = np.random.default_rng(0)

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n) * scale

    lambda_val = 1e-3


    # In[ ]:

    if glmnet:
        G = GLMNet(lambda_values=[lambda_val],
                   fit_intercept=fit_intercept,
                   standardize=standardize)
        G.fit(X, Y)
    else:
        G = GLM(fit_intercept=fit_intercept,
                standardize=standardize)
        G.fit(X, Y)

    I = G.design_.unscaler_ @ G.design_.scaler_
    x = rng.standard_normal(I.shape[0])
    assert np.allclose(x, I @ x)
    
    coef = rng.standard_normal(p)
    intercept = rng.standard_normal()
    
    grad, resid = compute_grad(G,
                               intercept,
                               coef,
                               G.design_,
                               Y,
                               scaled_input=scaled_input,
                               scaled_output=scaled_output)

    eD, S, U = _effective_D(X,
                            fit_intercept=fit_intercept,
                            standardize=standardize)

    D_ = np.column_stack([np.ones(X.shape[0]), X])
    assert np.allclose(G.design_ @ np.identity(p+1), eD)
    
    # now check the scaled input / output

    if not scaled_input:
        fit = X @ coef + intercept
    else:
        fit = G.design_ @ np.hstack([intercept, coef])

    resid_by_hand = Y - fit
        
    if not scaled_output:
        grad_by_hand = X.T @ resid_by_hand
        int_by_hand = np.sum(resid_by_hand)
    else:
        g_ = G.design_.T @ resid_by_hand
        int_by_hand = g_[0]
        grad_by_hand = g_[1:]
        
    assert np.allclose(resid, resid_by_hand)
    assert np.allclose(grad[1:], grad_by_hand)
    assert np.allclose(grad[0], np.sum(resid_by_hand))


def _effective_D(X,
                 fit_intercept,
                 standardize):

    centers = X.mean(0)
        
    X_c = X - centers[None,:]

    if standardize:
        scaling = np.sqrt((X_c**2).mean(0))
    else:
        scaling = np.ones(X.shape[1])

    D = np.column_stack([np.ones(X.shape[0]), X / scaling[None,:]])

    S = ScaleOperator(centers=centers,
                      scaling=scaling)
    U = UnscaleOperator(centers=centers,
                        scaling=scaling)

    if not fit_intercept:
        centers *= 0
        
    c = np.hstack([0, centers])
    s = np.hstack([1, scaling])
    rank1 = np.multiply.outer(np.ones(X.shape[0]), c / s)
    return D - rank1, S, U
