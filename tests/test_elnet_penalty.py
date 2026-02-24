import pytest
import numpy as np
import scipy.sparse

from glmnet.elnet import ElNet

def test_small_penalty(standardize,
                       fit_intercept,
                       n=50, p=10):

    rng = np.random.default_rng(0)
    X  = rng.standard_normal((n, p))
    X_s = scipy.sparse.csc_array(X)

    y = rng.standard_normal(n)
    W = rng.uniform(0, 1, size=(n,))

    spec = ElNet(lambda_val=0.5,
                 standardize=standardize,
                 fit_intercept=fit_intercept)

    spec.fit(X, y, sample_weight=W)
    beta = spec.raw_coef_.copy()
    out_s = spec.fit(X_s, y, sample_weight=W)
    beta_s = spec.raw_coef_.copy()

    assert np.allclose(beta, beta_s, rtol=1e-3, atol=1e-3)
