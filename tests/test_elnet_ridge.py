import pytest
import numpy as np
import scipy.sparse

from glmnet.elnet import ElNet

def test_simple_ridge(standardize,
                      fit_intercept,
                      n=50, p=10):

    rng = np.random.default_rng(0)
    X  = rng.standard_normal((n, p))
    X_s = scipy.sparse.csc_array(X)

    y = rng.standard_normal(n)
    W = rng.uniform(0, 1, size=(n,))

    spec = ElNet(lambda_val=0, standardize=standardize,
                 fit_intercept=fit_intercept)

    spec.fit(X, y, sample_weight=W)
    beta = spec.raw_coef_.copy()
    intercept = spec.raw_intercept_
    out_s = spec.fit(X_s, y, sample_weight=W)
    beta_s = spec.raw_coef_.copy()

    assert np.allclose(beta, beta_s, rtol=1e-3, atol=1e-3)

    X_np = X.copy()

    xm = (X_np.T @ W / W.sum()).reshape(-1)
    xm2 = (X_np**2).T @ W / W.sum()
    xs = np.sqrt(xm2 - xm**2).reshape(-1)

    if standardize:
        X_np /= xs[None,:]
        xm = (X_np.T @ W / W.sum()).reshape(-1)
        xm2 = (X_np**2).T @ W / W.sum()
        xs = np.sqrt(xm2 - xm**2).reshape(-1)
        if fit_intercept:
            X_np -= (xm/xs)[None,:]  
    else:
        xs = np.ones(p)
        if fit_intercept:
            xm = (X_np.T @ W / W.sum()).reshape(-1)
            X_np -= xm[None,:]

    if fit_intercept:
        X_eff = np.concatenate([np.ones((n, 1)), X_np], axis=1)
        Q = X_eff.T @ (W[:, None] * X_eff)
        beta_wls = np.linalg.solve(Q, X_eff.T @ (W * y))
        int_wls = beta_wls[0]
        beta_wls = beta_wls[1:]
    else:
        X_eff = X_np
        Q = X_eff.T @ (W[:, None] * X_eff)
        beta_wls = np.linalg.solve(Q, X_eff.T @ (W * y))
        int_wls = 0
        
    assert np.allclose(beta_wls, beta, rtol=1e-3, atol=1e-3)
    assert np.fabs((int_wls - intercept) / max(intercept, 1)) < 1e-3

