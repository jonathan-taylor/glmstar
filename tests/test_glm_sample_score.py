"""
Tests of GLMFamilySpec.sample_score.
"""
import pytest

import numpy as np
import statsmodels.api as sm

from glmnet.family import GLMFamilySpec, BinomFamilySpec
from glmnet.glm import GLMState

links = sm.families.links
FAMILIES = {'gaussian': sm.families.Gaussian(),
            'binomial': sm.families.Binomial(),
            'probit': sm.families.Binomial(link=links.Probit()),
            'poisson': sm.families.Poisson(),
            'gamma': sm.families.Gamma(link=links.Log()),
            'inverse_gaussian': sm.families.InverseGaussian(link=links.Log()),
            'negative_binomial': sm.families.NegativeBinomial(alpha=0.5)}


def _setup(name, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    eta = 0.2 + X @ np.array([0.5, -0.3])
    return GLMFamilySpec(base=FAMILIES[name]), X, eta


def _information(family, X, eta, w):
    state = GLMState(np.zeros(X.shape[1]), 0)
    state.mu = state.mean_parameter = family.base.link.inverse(eta)
    state.link_parameter = eta
    W = family.information(state, w) @ np.ones(X.shape[0])
    return X.T @ (W[:, None] * X)


@pytest.mark.parametrize('name', list(FAMILIES))
def test_shape_and_seed(name):
    family, X, eta = _setup(name)
    n = X.shape[0]
    draws = family.sample_score(eta, n_draws=7, rng=3)
    assert draws.shape == (n, 7)
    assert np.all(np.isfinite(draws))
    np.testing.assert_array_equal(draws, family.sample_score(eta, n_draws=7, rng=3))
    assert not np.allclose(draws, family.sample_score(eta, n_draws=7, rng=4))
    assert family.sample_score(eta, rng=3).shape == (n,)


@pytest.mark.parametrize('name', list(FAMILIES))
@pytest.mark.parametrize('dispersion', [1., 2.])
def test_covariance_is_information(name, dispersion):
    if dispersion != 1 and name in ('binomial', 'probit', 'poisson', 'negative_binomial'):
        pytest.skip('no dispersion')
    family, X, eta = _setup(name)
    n = X.shape[0]
    w = np.random.default_rng(1).integers(1, 4, n).astype(float)
    draws = X.T @ family.sample_score(eta, n_draws=6000, sample_weight=w,
                                      dispersion=dispersion, rng=1)
    target = dispersion * _information(family, X, eta, w)
    np.testing.assert_allclose(draws.mean(1), 0, atol=4 * np.sqrt(np.diag(target) / 6000).max())
    np.testing.assert_allclose(np.cov(draws), target, rtol=0.08, atol=0.08 * np.diag(target).max())


def test_logistic_is_bernoulli_minus_mean():
    family, X, eta = _setup('binomial')
    pi = 1 / (1 + np.exp(-eta))
    r = family.sample_score(eta, n_draws=50, rng=0)
    y = r + pi[:, None]
    np.testing.assert_allclose(y, np.round(y), atol=1e-12)
    assert set(np.unique(np.round(y))) <= {0., 1.}


def test_binom_family_spec_inherits():
    family = BinomFamilySpec()
    assert family.sample_score(np.zeros(5), rng=0).shape == (5,)


def test_errors():
    family, X, eta = _setup('poisson')
    with pytest.raises(ValueError):
        family.sample_score(eta, sample_weight=np.full(eta.shape, 0.5))
    with pytest.raises(ValueError):
        family.sample_score(eta, dispersion=2.)
    # non-integer weights are fine for a family that takes var_weights
    gamma, _, _ = _setup('gamma')
    assert gamma.sample_score(eta, sample_weight=np.full(eta.shape, 0.5), rng=0).shape == eta.shape
    tweedie = GLMFamilySpec(base=sm.families.Tweedie(link=links.Log(), var_power=1.5))
    with pytest.raises(ValueError):
        tweedie.sample_score(np.zeros(5))
