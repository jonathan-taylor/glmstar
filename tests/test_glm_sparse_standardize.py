"""
An unpenalized GLM fit to a sparse X with `standardize=True`: the Newton step
fits the raw sparse matrix and must put its coefficients on the design's
scale. It used to ignore the scaling, so the fit was wrong.
"""
import numpy as np
import pytest
import scipy.sparse
import statsmodels.api as sm

from glmnet import GLM
from glmnet.glm import GLMControl

TIGHT = GLMControl(epsnr=1e-12, mxitnr=100)
rng = np.random.default_rng(0)
n = 300
X = rng.standard_normal((n, 4)) * [1, 5, 0.1, 3] + [2, -1, 0, 4]
eta = 0.2 + 0.3 * X[:, 0] + 0.05 * X[:, 1]

FAMILIES = {
    'gaussian': (sm.families.Gaussian(), eta + rng.standard_normal(n)),
    'binomial': (sm.families.Binomial(), rng.binomial(1, 1 / (1 + np.exp(-eta)))),
    'poisson': (sm.families.Poisson(), rng.poisson(np.exp(eta))),
}


@pytest.mark.parametrize('name', list(FAMILIES))
def test_sparse_standardize(name):
    family, y = FAMILIES[name]
    y = y.astype(float)
    ref = GLM(family=family, control=TIGHT).fit(X, y)
    for standardize in [False, True]:
        for X_fit in [X, scipy.sparse.csc_array(X)]:
            g = GLM(family=family, standardize=standardize, control=TIGHT).fit(X_fit, y)
            np.testing.assert_allclose(g.coef_, ref.coef_, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(g.intercept_, ref.intercept_, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(g.deviance_, ref.deviance_, rtol=1e-10)
