import pytest

import numpy as np
import pandas as pd
from glmnet import GLM

rng = np.random.default_rng(0)

def test_gaussian(ridge_coef,
                  fit_intercept,
                  standardize,
                  sample_weight):

    n, p = 100, 20

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    W = sample_weight(n)
    
    # ridge_coef is divided by W.sum() below as the ridge term
    # is understood to be added to the weight-normalized likelihood

    # hence, below the ridge_coef in the quadratic form is not normazlized
    
    G = GLM(fit_intercept=fit_intercept,
            standardize=standardize,
            ridge_coef=ridge_coef / W.sum(), 
            weight_id='weight',
            response_id='Y')
    df = pd.DataFrame({'Y':Y, 'weight':W})
    
    G.fit(X, df)

    sqrtW = np.sqrt(W)

    D = G.design_
    effX = D @ np.identity(D.shape[1])
    effX = np.sqrt(W)[:,None] * effX
    effR = np.sqrt(W) * Y
    
    diag = np.ones(D.shape[1])
    diag[0] = 0

    # here we enter ridge_coef without scaling it

    effQ = effX.T @ effX + ridge_coef * np.diag(diag)
    
    Z = (effX.T @ effR)
    if not fit_intercept:
        effQ = effQ[1:,1:]
        Z = Z[1:]

    soln = np.linalg.solve(effQ, Z)

    if fit_intercept:
        coef = soln[1:]
        intercept = soln[0]
    else:
        coef = soln
        intercept = 0
        
    # (coef, intercept) are the scaled versions, need to transform to raw

    stack = np.hstack([intercept, coef])
    raw = D.unscaler_ @ stack
    int_final = raw[0]
    coef_final = raw[1:]
    
    assert np.allclose(int_final, G.intercept_)
    assert np.allclose(coef_final, G.coef_)


