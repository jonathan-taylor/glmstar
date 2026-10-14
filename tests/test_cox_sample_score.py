"""
Tests of CoxFamilySpec.sample_score, a thin wrapper of coxdev's
CoxDeviance.sample_score (whose own tests check each method's covariance).
"""
import pytest

import numpy as np
import pandas as pd

from glmnet.cox import CoxFamilySpec

coxdev = pytest.importorskip('coxdev')
if not hasattr(coxdev.CoxDeviance, 'sample_score'):
    pytest.skip('needs coxdev with CoxDeviance.sample_score', allow_module_level=True)


def _family(n=300, start=False, strata=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    eta = X @ np.array([0.5, -0.3])
    T = rng.exponential(size=n) / np.exp(eta)
    C = rng.exponential(2, size=n)
    event_data = pd.DataFrame({'event': np.ceil(np.minimum(T, C) * 5) / 5 + 0.2,
                               'status': (T <= C).astype(int)})
    kw = {}
    if start:
        event_data['start'] = rng.uniform(0, 0.1, size=n)
        kw['start_id'] = 'start'
    if strata:
        event_data['strata'] = rng.choice(['a', 'b', 'c'], size=n)
        kw['strata_id'] = 'strata'
    family = CoxFamilySpec(event_data,
                           event_id='event',
                           status_id='status',
                           tie_breaking='breslow',
                           **kw)
    return family, X, eta


@pytest.mark.parametrize('method', CoxFamilySpec.sample_score_methods)
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('strata', [False, True])
def test_shape_and_seed(method, start, strata):
    family, X, eta = _family(start=start, strata=strata)
    n = X.shape[0]

    draws = family.sample_score(eta, method=method, n_draws=7, rng=3)
    assert draws.shape == (n, 7)
    assert np.all(np.isfinite(draws))
    np.testing.assert_array_equal(draws, family.sample_score(eta, method=method, n_draws=7, rng=3))
    assert not np.allclose(draws, family.sample_score(eta, method=method, n_draws=7, rng=4))
    assert family.sample_score(eta, method=method, rng=3).shape == (n,)


def test_bad_method():
    family, X, eta = _family()
    with pytest.raises(ValueError):
        family.sample_score(eta, method='poisson')


@pytest.mark.parametrize('strata', [False, True])
def test_multinomial_covariance_is_information(strata):
    family, X, eta = _family(strata=strata)
    n = X.shape[0]
    draws = X.T @ family.sample_score(eta, method='multinomial', n_draws=6000, rng=1)
    I = family._coxdev.information(eta, np.ones(n))
    target = X.T @ np.column_stack([I @ X[:, j] for j in range(X.shape[1])])
    np.testing.assert_allclose(np.cov(draws), target, rtol=0.08, atol=0.08 * np.diag(target).max())
