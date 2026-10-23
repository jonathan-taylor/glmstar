"""
REML and ML with the binomial's other links: statsmodels derives Probit,
CLogLog, LogLog and Cauchy from Logit, and the criterion took them for the
logit (`glmnet.gam.reml._jax_family`).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm

pytest.importorskip('jax')

from glmnet.gam import GAM, cubic_regression_spline
from glmnet.gam.reml import _jax_family

links = sm.families.links


@pytest.mark.parametrize('link', [links.Logit(), links.Probit(), links.CLogLog()])
def test_inverse_link(link):
    import jax
    linkinv = _jax_family(sm.families.Binomial(link=link))[0]
    eta = np.linspace(-3, 3, 13)
    with jax.enable_x64(True):   # (as the criterion)
        np.testing.assert_allclose(np.asarray(linkinv(eta)), link.inverse(eta), rtol=1e-12)


@pytest.mark.parametrize('link', [links.Cauchy(), links.LogLog()])
def test_unsupported_links_raise(link):
    with pytest.raises(NotImplementedError, match='not supported'):
        _jax_family(sm.families.Binomial(link=link))


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_probit_against_mgcv(method):
    pytest.importorskip('rpy2')
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    try:
        ro.r('suppressMessages(library(mgcv))')
    except Exception:
        pytest.skip('mgcv not available')
    rng = np.random.default_rng(0)
    n = 600
    x = rng.uniform(0, 1, n)
    D = pd.DataFrame({'x': x, 'y': rng.binomial(1, norm.cdf(np.sin(6 * x))).astype(float)})
    family = sm.families.Binomial(link=links.Probit())
    with pytest.warns():
        G = GAM(terms=[cubic_regression_spline('x', k=10)], family=family, method=method).fit(D, D['y'])
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['D'] = D
    edf, score = ro.r(f'm <- gam(y ~ s(x, bs="cr", k=10), family=binomial(link="probit"), '
                      f'data=D, method="{method}"); c(sum(m$edf), m$gcv.ubre)')
    np.testing.assert_allclose(float(G.smoothing_.score), score, rtol=1e-7)
    # (mgcv's edf counts the intercept)
    np.testing.assert_allclose(float(np.sum(G.edf_)) + 1, edf, rtol=1e-5)
