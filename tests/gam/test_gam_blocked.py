"""
REML and ML with one large random effect, factored by blocks
(`glmnet.gam.blocked`): the blocked Hessian's operations against the dense
one's, the criterion and its gradient, and the fits (smoothing parameters,
criterion, edf) of the blocked backend against the dense one; the choice of
the backend; and a random effect of 5 * 10^4 levels without dense matrices.
"""
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
jax = pytest.importorskip('jax')

from glmnet.gam import GAM, GLMM, GAMControl, s, re, linear, factor
from glmnet.gam.blocked import SplitDesign, SplitPenalty, BlockedHessian
from glmnet.gam.linalg import DenseHessian
from glmnet.gam.reml import LaplaceCriterion

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


def _split_problem(seed=0, n=400, R=30, m=6):
    rng = np.random.default_rng(seed)
    P = 1 + R + m - 1
    a_start = 2
    a = np.arange(a_start, a_start + R)
    e_idx = np.setdiff1d(np.arange(P), a)
    codes = rng.integers(0, R, n)
    values = rng.uniform(0.5, 2, n)
    E = rng.standard_normal((n, m))
    X = SplitDesign(P, a_start, R, codes, values, e_idx, E)
    Xd = np.zeros((n, P))
    Xd[np.arange(n), a[codes]] = values
    Xd[:, e_idx] = E
    h = rng.uniform(0.2, 2, n)
    S = SplitPenalty(P, e_idx)
    Sd = np.zeros((P, P))
    lam_a = rng.uniform(0.5, 2, R)
    S.add_diagonal(a, lam_a)
    Sd[a, a] += lam_a
    block = e_idx[2:5]
    U = rng.standard_normal((3, 3))
    U = U @ U.T
    S.add_block(block, U)
    Sd[np.ix_(block, block)] += U
    return X, Xd, h, S, Sd, a, e_idx, block, U


def test_operations():
    X, Xd, h, S, Sd, a, e_idx, block, U = _split_problem()
    B, D = BlockedHessian(X, h, S), DenseHessian(Xd, h, Sd)
    np.testing.assert_allclose(B.logdet(), D.logdet(), rtol=1e-12)
    rng = np.random.default_rng(1)
    r = rng.standard_normal((X.P, 3))
    np.testing.assert_allclose(B.solve(r), D.solve(r), rtol=1e-9, atol=1e-12)
    d = rng.uniform(size=len(a))
    np.testing.assert_allclose(B.trace(a, d), D.trace(a, d), rtol=1e-10)
    np.testing.assert_allclose(B.trace(block, U), D.trace(block, U), rtol=1e-10)
    groups = e_idx[1:5].reshape((2, 2))
    Bq = np.array([[1., .3], [.3, 2.]])
    np.testing.assert_allclose(B.trace_blocks(groups, Bq), D.trace_blocks(groups, Bq), rtol=1e-10)
    np.testing.assert_allclose(B.leverages(), D.leverages(), rtol=1e-9)
    # the design's products, and the edf: 1 - diag(H^{-1} S)
    beta = rng.standard_normal(X.P)
    v = rng.standard_normal(Xd.shape[0])
    np.testing.assert_allclose(X @ beta, Xd @ beta, rtol=1e-12)
    np.testing.assert_allclose(X.T @ v, Xd.T @ v, rtol=1e-12)
    np.testing.assert_allclose(S @ beta, Sd @ beta, rtol=1e-12)
    np.testing.assert_allclose(beta @ S @ beta, beta @ Sd @ beta, rtol=1e-12)
    np.testing.assert_allclose(B.edf(S), 1 - np.diag(np.linalg.solve(D._H, Sd)), atol=1e-10)
    # ML: restricted to the penalized coordinates (all of the large block)
    coords = np.sort(np.concatenate([a, block]))
    Br, Dr = B.restrict(coords), D.restrict(coords)
    np.testing.assert_allclose(Br.logdet(), Dr.logdet(), rtol=1e-12)
    np.testing.assert_allclose(Br.leverages(), Dr.leverages(), rtol=1e-9)
    np.testing.assert_allclose(Br.trace(a, d), Dr.trace(a, d), rtol=1e-10)
    np.testing.assert_allclose(Br.solve(r), Dr.solve(r), rtol=1e-9, atol=1e-12)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n, G = 1500, 60
    D = pd.DataFrame({'g': rng.integers(0, G, n), 'x': rng.standard_normal(n),
                      'u': rng.uniform(size=n), 'c': rng.choice(list('abc'), n),
                      'v': rng.uniform(0, 2, n)})
    b = rng.normal(0, .7, G)
    D['y'] = 0.3 * D['x'] + np.sin(3 * D['u']) + b[D['g']] + rng.standard_normal(n)
    D['yp'] = rng.poisson(np.exp(0.3 + 0.3 * D['x'] + 0.5 * np.sin(3 * D['u']) + 0.5 * b[D['g']]))
    return D


FAMILIES = {'gaussian': (sm.families.Gaussian(), 'y'), 'poisson': (sm.families.Poisson(), 'yp')}


def _fit(data, family, method, backend):
    fam, y = FAMILIES[family]
    terms = [linear('x'), s('u'), factor('c'), re('g'), re('g', by='v')]
    return GLMM(terms, family=fam, method=method,
               control=GAMControl(reml_backend=backend, **TIGHT)).fit(data, data[y])


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_criterion(data, family, method):
    Gd, Gb = _fit(data, family, method, 'dense'), _fit(data, family, method, 'blocked')
    with jax.enable_x64(True):
        Cd = LaplaceCriterion(Gd, [1, 3, 4], method=method)
        Cb = LaplaceCriterion(Gb, [1, 3, 4], method=method)
        assert Cb.blocked and not Cd.blocked
        x = Cd.start() + np.random.default_rng(1).uniform(-0.5, 0.5, Cd.n_params)
        vd, gd = Cd.value_and_grad(x)
        vb, gb = Cb.value_and_grad(x)
    np.testing.assert_allclose(vb, vd, rtol=1e-12)
    np.testing.assert_allclose(gb, gd, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_fits(data, family, method):
    Gd, Gb = _fit(data, family, method, 'dense'), _fit(data, family, method, 'blocked')
    assert Gb.smoothing_.converged and Gd.smoothing_.converged
    # (the inner fit by backfitting, polished)
    assert type(Gb.regularizer_).__name__ == 'GAMRegularizer'
    np.testing.assert_allclose(Gb.reml_score_, Gd.reml_score_, rtol=1e-11)
    np.testing.assert_allclose(Gb.smoothing_.lam, Gd.smoothing_.lam, rtol=1e-5)
    np.testing.assert_allclose(Gb.state_.link_parameter, Gd.state_.link_parameter, atol=1e-8)
    # mgcv's edf, from the blocked factorization
    np.testing.assert_allclose(Gb.edf_.values, Gd.edf_.values, atol=1e-8)
    np.testing.assert_allclose(Gb.variance_components_, Gd.variance_components_, rtol=1e-5)


def test_backend_choice(data):
    # auto: blocked above max_dense_coef, with one random effect taking the excess
    G = GLMM([linear('x'), re('g')], method='REML',
            control=GAMControl(max_dense_coef=20)).fit(data, data['y'])
    H = GLMM([linear('x'), re('g')], method='REML').fit(data, data['y'])
    assert type(G.regularizer_).__name__ == 'GAMRegularizer'
    assert G.smoothing_.edf is not None and H.smoothing_.edf is None
    np.testing.assert_allclose(G.reml_score_, H.reml_score_, rtol=1e-10)
    # two large random effects: not blocked (matrix-free is to come)
    D = data.assign(h=np.random.default_rng(2).integers(0, 50, len(data)))
    with pytest.raises(ValueError, match='not one large one'):
        GLMM([re('g'), re('h')], method='REML',
            control=GAMControl(max_dense_coef=40)).fit(D, D["y"])
    with pytest.raises(ValueError, match='needs a random effect'):
        GLMM([s('u')], method='REML', control=GAMControl(reml_backend='blocked')).fit(data, data['y'])


@pytest.mark.large
def test_large():
    # 5 * 10^4 levels: REML without the dense 5 * 10^4 square matrices
    rng = np.random.default_rng(3)
    n, G = 150_000, 50_000
    D = pd.DataFrame({'g': rng.integers(0, G, n), 'x': rng.standard_normal(n),
                      'u': rng.uniform(size=n)})
    b = rng.normal(0, 0.5, G)
    D['y'] = 0.3 * D['x'] + np.sin(3 * D['u']) + b[D['g']] + rng.standard_normal(n)
    tracemalloc.start()
    try:
        M = GLMM([linear('x'), s('u'), re('g')], method='REML').fit(D, D['y'])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # (a dense H would be 20 GB)
    assert peak < 3e9
    assert M.smoothing_.converged
    vc = M.variance_components_['sd']
    np.testing.assert_allclose(vc['re(g)'], 0.5, rtol=0.05)
    np.testing.assert_allclose(vc['scale'], 1.0, rtol=0.02)
    assert M.edf_.notna().all()
