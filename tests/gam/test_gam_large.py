"""
Large random effects without dense matrices: a random effect of 10^5
levels is fitted by backfitting, with `fixed_effects_`, `summary()`,
predictions and plots, without forming its R x R penalty or the N x P
design (the memory traced by `tracemalloc`, which numpy reports to). The
dense quantities (the posterior covariance, standard errors, REML's
Hessian) are not formed above `GAMControl(max_dense_coef=...)`: they are
None, or raise saying so.
"""
import tracemalloc

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GLMM, GAMControl, s, random_effect as re, linear

N, R = 200_000, 100_000


@pytest.fixture(scope='module')
def big():
    rng = np.random.default_rng(0)
    g = rng.integers(0, R, N)
    D = pd.DataFrame({'g': g, 'x': rng.standard_normal(N), 'u': rng.uniform(0, 1, N)})
    b = rng.normal(0, 0.5, R)
    D['y'] = 1 + 0.3 * D['x'] + np.sin(3 * D['u']) + b[g] + rng.standard_normal(N)
    return D


def _peak(f):
    tracemalloc.start()
    try:
        out = f()
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_fit_without_dense_matrices(big):
    def run():
        G = GAM(terms=[linear('x'), re('g', lam=1.)]).fit(big, big['y'])
        return G, G.fixed_effects_, G.summary(), G.predict(big.iloc[:100])
    (G, fe, summary, pred), peak = _peak(run)
    # (a dense R x R penalty would be 80 GB; the N x P design 160 GB)
    assert peak < 1e9
    assert G.converged_
    np.testing.assert_allclose(fe.loc['linear(x)', 'estimate'], 0.3, atol=0.02)
    assert np.all(fe['std_error'] > 0)
    assert 'edf_influence' not in summary.columns
    assert pred.shape == (100,)
    # the dense quantities: not formed, and saying why
    assert G.posterior_covariance_ is None and G.edf_influence_ is None
    with pytest.raises(ValueError, match='max_dense_coef'):
        G.predict(big.iloc[:5], return_std=True)


def test_plot_without_bands(big):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    G = GAM(terms=[s('u', lam=1e-3), re('g', lam=1.)]).fit(big, big['y'])
    _, peak = _peak(lambda: G.plot(terms=[0]))
    assert peak < 1e9
    # (no covariance: the curve without its bands)
    with pytest.raises(ValueError, match='max_dense_coef'):
        G.partial_effect(0, np.linspace(0, 1, 5), return_std=True)


def test_reml_refuses(big):
    # (one large random effect is factored by blocks, glmnet.gam.blocked; two
    # are not yet)
    small = big.iloc[:30_000].assign(h=lambda d: (d['g'] * 7919) % 25_000)
    with pytest.raises(ValueError, match='max_dense_coef'):
        GLMM(terms=[linear('x'), re('g'), re('h')], method='REML').fit(small, small['y'])


def test_threshold(big):
    # below the threshold, the covariance as before; above, none
    part = big.iloc[:2000].assign(g=lambda d: d['g'] % 50)
    G = GAM(terms=[linear('x'), re('g', lam=1.)]).fit(part, part['y'])
    assert G.posterior_covariance_.shape == (52, 52)
    H = GAM(terms=[linear('x'), re('g', lam=1.)],
            control=GAMControl(max_dense_coef=40)).fit(part, part['y'])
    assert H.posterior_covariance_ is None
    np.testing.assert_allclose(H.predict(part), G.predict(part))
    # the fixed effects' standard errors do not need it (backfitting)
    np.testing.assert_allclose(H.fixed_effects_, G.fixed_effects_, rtol=1e-6)
