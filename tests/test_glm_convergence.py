import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm

import pytest

from glmnet import GLM
from glmnet.glm import GaussianGLM
from glmnet.cox import CoxLM, CoxFamily

links = sm.families.links

# GLM at the default control matches statsmodels for non-canonical links
# (it started at the intercept 0, outside the domain of e.g. the inverse
# link, and did not halve steps leaving it) and near separation

rng = np.random.default_rng(0)
n = 200
X = rng.uniform(0, 1, (n, 2))
mu = 1 + X @ [1, 2]
y_gamma = rng.gamma(5, mu / 5)
y_count = rng.poisson(mu)

CASES = {'gamma-inverse': (sm.families.Gamma(), y_gamma),
         'gamma-identity': (sm.families.Gamma(links.Identity()), y_gamma),
         'gamma-log': (sm.families.Gamma(links.Log()), y_gamma),
         'inverse-gaussian-log': (sm.families.InverseGaussian(links.Log()), rng.wald(mu, 5)),
         'poisson-identity': (sm.families.Poisson(links.Identity()), y_count),
         'poisson-sqrt': (sm.families.Poisson(links.Sqrt()), y_count),
         'tweedie': (sm.families.Tweedie(var_power=1.5, link=links.Log()), y_gamma),
         'binomial-cloglog': (sm.families.Binomial(links.CLogLog()),
                              rng.binomial(1, 1 / (1 + np.exp(-(X @ [1, -2])))))}


@pytest.mark.parametrize('name', sorted(CASES))
def test_links(name):
    family, y = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reference = sm.GLM(y, sm.add_constant(X), family=family).fit(tol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fit = GLM(family=family, summarize=True).fit(X, y)
    np.testing.assert_allclose(fit.params_.values, reference.params, atol=1e-5)


def test_near_separation():
    # ISLP's high earners: a logistic fit with a rare response and large
    # coefficients, where epsnr=1e-6 stopped in the 4th significant figure
    ISLP = pytest.importorskip('ISLP')
    from ISLP.models import ModelSpec, poly
    Wage = ISLP.load_data('Wage')
    y = np.asarray(Wage['wage'] > 250, float)
    Z = ModelSpec([poly('age', degree=4)]).fit_transform(Wage)
    reference = sm.GLM(y, Z, family=sm.families.Binomial()).fit(tol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # the 'intercept' column is dropped
        fit = GLM(family=sm.families.Binomial()).fit(Z, y)
    np.testing.assert_allclose(fit.params_.values, reference.params, rtol=1e-7)


# a constant column of X other than 'intercept' is excluded (it made the fit singular)

@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('family', [sm.families.Gaussian(), sm.families.Poisson()])
def test_constant_column(family, sparse):
    x = rng.uniform(0, 2, n)
    Z = np.power.outer(x, np.arange(3))           # a column of ones first
    y = rng.poisson(np.exp(0.5 + 0.3 * x))
    reference = sm.GLM(y, Z, family=family).fit(tol=1e-14)
    Zfit = scipy.sparse.csc_array(Z) if sparse else Z
    with pytest.warns(UserWarning, match='constant column'):
        fit = GLM(family=family, summarize=True).fit(Zfit, y)
    assert fit.coef_[0] == 0
    np.testing.assert_allclose(fit.params_.values, reference.params, rtol=1e-6)
    np.testing.assert_allclose(fit.predict(Zfit), reference.fittedvalues, rtol=1e-6)
    assert len(fit.summary_) == 3


def test_constant_column_dataframe():
    x = rng.uniform(0, 2, n)
    Z = pd.DataFrame({'one': np.ones(n), 'x': x})
    y = 1 + x + rng.standard_normal(n)
    with pytest.warns(UserWarning, match=r"constant column\(s\) \['one'\]"):
        fit = GaussianGLM().fit(Z, y)
    reference = sm.OLS(y, Z).fit()
    np.testing.assert_allclose(fit.params_.values, reference.params, rtol=1e-8)
    # without an intercept the constant column is the intercept
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fit0 = GaussianGLM(fit_intercept=False).fit(Z, y)
    np.testing.assert_allclose(fit0.coef_, reference.params, rtol=1e-8)


def test_constant_column_cox():
    x = rng.standard_normal(n)
    t = rng.exponential(np.exp(-x))
    Y = pd.DataFrame({'event': t, 'status': np.ones(n)})
    Z = np.column_stack([x, np.full(n, 2.)])
    family = CoxFamily(tie_breaking='breslow')
    with pytest.warns(UserWarning, match='constant column'):
        fit = CoxLM(family=family, summarize=True).fit(Z, Y)
    alone = CoxLM(family=family).fit(Z[:, :1], Y)
    np.testing.assert_allclose(fit.coef_, [alone.coef_[0], 0])
    assert len(fit.summary_) == 1
