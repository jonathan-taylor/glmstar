import numpy as np

from glmnet import LogNet


def test_lognet_df_max_coefs():
    # with df_max set the solver's buffer holds one block of
    # nx = min(2 * df_max + 20, p) < p coefficients per lambda, and
    # _extract_fits used to reshape it with blocks of p
    rng = np.random.default_rng(0)
    n, p, df_max = 300, 100, 5
    X = rng.standard_normal((n, p))
    y = (X[:, :5] @ (3 * rng.standard_normal(5)) + rng.standard_normal(n) > 0).astype(float)

    L = LogNet(nlambda=30, df_max=df_max).fit(X, y)
    assert L._args['nx'] < p
    full = LogNet(nlambda=30).fit(X, y)
    k = L.coefs_.shape[0]
    np.testing.assert_allclose(L.coefs_, full.coefs_[:k], rtol=1e-6, atol=1e-10)
    assert np.all((np.abs(L.coefs_) > 0).sum(1) <= df_max + 1)
