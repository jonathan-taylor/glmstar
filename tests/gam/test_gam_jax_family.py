"""
A family of one's own, in JAX (`glmnet.gam.JaxFamily`, and any family with
`jax_pieces()`): the probit by hand against statsmodels' probit, for a
`GAM` of fixed smoothing and a `GLMM` by REML and ML; a Gaussian by hand
(its dispersion free) against the Gaussian; a cauchit, which REML has no
built-in for; and the errors of a family missing its pieces.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

pytest.importorskip('scatter_smooth')
pytest.importorskip('jax')

import jax.numpy as jnp
from jax.scipy.special import xlogy
from jax.scipy.stats import norm

from glmnet.gam import (GAM, GLMM, GAMControl, JaxFamily, ExperimentalWarning, s,
                        random_effect as re)
from glmnet.gam.reml import _jax_family

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)
# the relative tolerance of the matches: the fits agree to rounding (the
# smoothing parameters to ~1e-14), the coefficients to the local scoring's
RTOL = 1e-8


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        yield


def bernoulli_deviance(y, mu):
    return 2 * (xlogy(y, y) - xlogy(y, mu) + xlogy(1 - y, 1 - y) - xlogy(1 - y, 1 - mu))


def probit():
    return JaxFamily(norm.cdf, bernoulli_deviance, name='probit')


def cauchit():
    return JaxFamily(lambda eta: 0.5 + jnp.arctan(eta) / np.pi, bernoulli_deviance,
                     name='cauchit')


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    n = 500
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'g': rng.integers(0, 15, n)})
    b = rng.normal(0, 0.6, 15)
    eta = np.sin(5 * D['x']) + b[D['g']]
    D['y'] = (rng.uniform(size=n) < stats.norm.cdf(eta)).astype(float)
    D['yc'] = (rng.uniform(size=n) < stats.cauchy.cdf(eta)).astype(float)
    D['yg'] = eta + 0.5 * rng.standard_normal(n)
    return D


def assert_close(a, b, rtol=RTOL):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=0)


def assert_same_fit(mine, builtin, D):
    mu, se = mine.predict(D, return_std=True)
    mu0, se0 = builtin.predict(D, return_std=True)
    assert_close(mu, mu0)
    assert_close(se, se0)
    assert_close(mine.edf_, builtin.edf_)
    assert_close(mine.deviance_, builtin.deviance_)
    assert_close(np.sqrt(np.diag(mine.posterior_covariance_)),
                 np.sqrt(np.diag(builtin.posterior_covariance_)))


def test_probit_gam(data):
    terms = [s('x', lam=0.01), re('g', lam=0.5)]
    builtin = sm.families.Binomial(link=sm.families.links.Probit())
    G0 = GAM(terms, family=builtin, control=GAMControl(**TIGHT)).fit(data, data['y'])
    G = GAM(terms, family=probit(), control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert_same_fit(G, G0, data)
    assert G.dispersion_ == 1


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_probit_glmm(data, method):
    terms = [s('x'), re('g')]
    builtin = sm.families.Binomial(link=sm.families.links.Probit())
    M0 = GLMM(terms, family=builtin, method=method,
              control=GAMControl(**TIGHT)).fit(data, data['y'])
    M = GLMM(terms, family=probit(), method=method,
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert M.smoothing_.converged and M0.smoothing_.converged
    assert_close(M.smoothing_.lam, M0.smoothing_.lam)
    # (the Bernoulli's saturated log-likelihood is 0: the scores the same)
    assert_close(M.smoothing_.score, M0.smoothing_.score)
    assert_same_fit(M, M0, data)


def test_builtin_probit_not_logit(data):
    # statsmodels derives Probit from Logit: REML's inverse link is the probit's
    linkinv = _jax_family(sm.families.Binomial(link=sm.families.links.Probit()))[0]
    np.testing.assert_allclose(linkinv(1.), stats.norm.cdf(1.))


def test_gaussian_free_dispersion(data):
    # the Gaussian by hand: a free dispersion, its saturated log-likelihood
    gaussian = JaxFamily(lambda eta: eta, lambda y, mu: (y - mu)**2,
                         saturated_loglike=lambda y, a, phi: jnp.sum(
                             -0.5 * jnp.log(2 * np.pi * phi / a)),
                         fixed_dispersion=False, name='gaussian')
    terms = [s('x'), re('g')]
    M0 = GLMM(terms, control=GAMControl(**TIGHT)).fit(data, data['yg'])
    M = GLMM(terms, family=gaussian, control=GAMControl(**TIGHT)).fit(data, data['yg'])
    assert_close(M.smoothing_.lam, M0.smoothing_.lam, rtol=1e-6)
    assert_close(M.smoothing_.dispersion, M0.smoothing_.dispersion, rtol=1e-6)
    assert_close(M.smoothing_.score, M0.smoothing_.score, rtol=1e-8)
    assert_close(M.predict(data), M0.predict(data), rtol=1e-6)
    # its log-likelihood, the Gaussian's
    y, mu = data['yg'].values, M.predict(data)
    assert_close(gaussian.loglike(y, mu, scale=0.3), sm.families.Gaussian().loglike(y, mu, scale=0.3))


def test_cauchit(data):
    # statsmodels has the cauchit (its Cauchy link) for the fit, but REML had none
    sm_cauchit = sm.families.Binomial(link=sm.families.links.Cauchy())
    terms = [s('x', lam=0.01)]
    G0 = GAM(terms, family=sm_cauchit, control=GAMControl(**TIGHT)).fit(data, data['yc'])
    G = GAM(terms, family=cauchit(), control=GAMControl(**TIGHT)).fit(data, data['yc'])
    assert_close(G.predict(data), G0.predict(data))
    with pytest.raises(NotImplementedError, match='JaxFamily'):
        GLMM([s('x'), re('g')], family=sm_cauchit).fit(data, data['yc'])
    M = GLMM([s('x'), re('g')], family=cauchit(),
             control=GAMControl(**TIGHT)).fit(data, data['yc'])
    assert M.smoothing_.converged
    assert np.max(np.abs(M.smoothing_.gradient)) < 1e-5
    assert np.all(np.isfinite(M.smoothing_.lam))


class CauchitBinomial(sm.families.Binomial):
    """statsmodels' binomial with the Cauchy link, and its pieces for REML."""

    def __init__(self):
        super().__init__(link=sm.families.links.Cauchy())

    def jax_pieces(self):
        return (lambda eta: 0.5 + jnp.arctan(eta) / np.pi,
                lambda y, mu, t: bernoulli_deviance(y, mu),
                lambda y, a, phi, t: 0. * phi,
                True)


def test_protocol(data):
    # a statsmodels family with jax_pieces(): REML uses them
    terms = [s('x'), re('g')]
    M0 = GLMM(terms, family=cauchit(), control=GAMControl(**TIGHT)).fit(data, data['yc'])
    M = GLMM(terms, family=CauchitBinomial(), control=GAMControl(**TIGHT)).fit(data, data['yc'])
    assert_close(M.smoothing_.lam, M0.smoothing_.lam, rtol=1e-6)
    assert_close(M.smoothing_.score, M0.smoothing_.score)
    assert_close(M.predict(data), M0.predict(data), rtol=1e-6)


def test_errors():
    with pytest.raises(TypeError, match='linkinv'):
        JaxFamily(None, bernoulli_deviance)
    with pytest.raises(TypeError, match='linkinv'):
        JaxFamily(norm.cdf, None)
    with pytest.raises(ValueError, match='saturated_loglike'):
        JaxFamily(lambda eta: eta, lambda y, mu: (y - mu)**2, fixed_dispersion=False)

    class Bad(sm.families.Binomial):
        def jax_pieces(self):
            return (norm.cdf, None)
    with pytest.raises(ValueError, match='jax_pieces'):
        _jax_family(Bad())

    class NoLink(object):
        pass
    with pytest.raises(NotImplementedError, match='jax_pieces'):
        _jax_family(NoLink())


def test_link_and_variance():
    # the link by bisection, and the variance of the deviance: the binomial's
    fam = probit()
    mu = np.array([1e-6, 0.1, 0.5, 0.9, 1 - 1e-6])
    # (to the precision of norm.cdf near 1)
    np.testing.assert_allclose(fam.link(mu), stats.norm.ppf(mu), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fam.variance(mu), mu * (1 - mu), rtol=1e-12)
    eta = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(fam.link.inverse_deriv(eta), stats.norm.pdf(eta), rtol=1e-12)
    np.testing.assert_allclose(fam.link.deriv(mu), 1 / stats.norm.pdf(stats.norm.ppf(mu)),
                               rtol=1e-8)
