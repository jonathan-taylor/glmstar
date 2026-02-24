import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from glmnet.regularized_glm import RegGLM

rng = np.random.default_rng(0)

def nonuniform_(n):
    W = rng.uniform(0, 1, size=(n,))
    W[:n//2] *= 2
    return W

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, nonuniform_])
def test_dense_vs_sparse_gaussnet_flex(standardize, fit_intercept, sample_weight, n=200, p=20):
    """
    Compare RegGLM fits for dense vs. sparse X (Gaussian family), as in the original flex test.
    """
    sample_weight = sample_weight(n)
    sample_weight = sample_weight / sample_weight.sum()

    X = rng.standard_normal((n, p))
    X[:, 4] *= 4.5
    X[:, 2] *= 1.3
    X[:, 0] *= 0.6
    X[:, 1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1, 2]
    y = 2 * rng.standard_normal(n) + X @ beta

    df = pd.DataFrame({'response': y, 'weight': sample_weight})
    lambda_val = 2 / np.sqrt(n)

    # Dense fit
    G_dense = RegGLM(lambda_val=lambda_val,
                     alpha=1.0,
                     standardize=standardize,
                     fit_intercept=fit_intercept,
                     response_id='response',
                     weight_id='weight')
    G_dense.fit(X, df)
    yhat_dense = G_dense.design_ @ np.hstack([G_dense.intercept_, G_dense.coef_])

    # Sparse fit
    X_sparse = scipy.sparse.csc_array(X)
    G_sparse = RegGLM(lambda_val=lambda_val,
                      alpha=1.0,
                      standardize=standardize,
                      fit_intercept=fit_intercept,
                      response_id='response',
                      weight_id='weight')
    G_sparse.fit(X_sparse, df)
    yhat_sparse = G_sparse.design_ @ np.hstack([G_sparse.intercept_, G_sparse.coef_])

    # Compare
    assert np.allclose(yhat_dense, yhat_sparse, atol=1e-5, rtol=1e-4)
    assert np.isclose(G_dense.intercept_, G_sparse.intercept_, atol=1e-5)
    assert np.allclose(G_dense.coef_, G_sparse.coef_, atol=1e-5, rtol=1e-4) 