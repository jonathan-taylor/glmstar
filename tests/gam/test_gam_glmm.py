"""
`GLMM` against `GAM`: a GAM's smoothing is fixed, a GLMM's estimated.
`GLMM(method=...)` and `GLMM(method='fixed')` are deprecated (a
FutureWarning, the same fit); and a GLMM with some smoothing parameters held
fixed (a term's `lam` a list with None for those to estimate), against the
GAM at the estimates and the GLMM estimating them all.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')
pytest.importorskip('jax')

from glmnet.gam import (GAM, GLMM, GAMControl, ExperimentalWarning, s, re, categorical,
                        tensor_spline as te, varying, cubic_regression_spline)
from glmnet.gam.selection import Laplace

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        yield


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 400
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(-1, 2, n),
                      'g': rng.integers(0, 12, n), 'c': rng.choice(list('ab'), n)})
    b = rng.normal(0, 0.5, 12)
    D['y'] = (np.sin(3 * D['x']) * D['z'] + b[D['g']] + 0.4 * (D['c'] == 'b') +
              0.3 * rng.standard_normal(n))
    return D


def test_gam_method_fixed_deprecated(data):
    terms = [s('x', df=4), re('g', lam=0.1)]
    G = GAM(terms).fit(data, data['y'])
    with pytest.warns(FutureWarning, match="method='fixed'"):
        H = GAM(terms).fit(data, data['y'])
    np.testing.assert_allclose(H.predict(data), G.predict(data))
    assert not hasattr(G, 'smoothing_')


@pytest.mark.parametrize('method', ['REML', 'ML', Laplace('REML')])
def test_gam_estimating_deprecated(data, method):
    terms = [s('x'), re('g')]
    G = GLMM(terms, method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    with pytest.warns(FutureWarning, match='GLMM'):
        H = GLMM(terms, method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(H.smoothing_.lam, G.smoothing_.lam)
    np.testing.assert_allclose(H.predict(data), G.predict(data))


def test_glmm(data):
    terms = [s('x'), re('g')]
    G = GLMM(terms, control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.method == 'REML' and G.smoothing_.method == 'REML'
    assert G.smoothing_.converged
    assert list(G.variance_components_.index) == ['s(x)', 're(g)', 'scale']
    # a GLMM's fixed smoothing is a GAM
    with pytest.warns(FutureWarning, match='GAM'):
        H = GLMM([s('x', df=4), re('g', lam=0.1)], method='fixed').fit(data, data['y'])
    np.testing.assert_allclose(H.predict(data),
                               GAM([s('x', df=4), re('g', lam=0.1)]).fit(data, data['y'])
                               .predict(data))
    # GAM's own warnings name the class
    with pytest.warns(ExperimentalWarning, match=r'GLMM\(method='):
        GLMM(terms).fit(data, data['y'])


def test_some_terms_fixed(data):
    # the spline's smoothing parameter given, the random effect's estimated
    G = GLMM([s('x', lam=1e-3), re('g')], control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.smoothing_.terms == ['re(g)']
    assert G.terms_[0].lam == 1e-3
    lam = G.smoothing_.lam[0]
    H = GAM([s('x', lam=1e-3), re('g', lam=lam)], control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(H.predict(data), G.predict(data), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_some_penalties_fixed(data, method):
    # one margin's smoothing parameter of a tensor product given, the other's
    # estimated
    fixed = 5e-3
    terms = [te('x', 'z', k=5, lam=[fixed, None]), categorical('c')]
    G = GLMM(terms, method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.smoothing_.converged
    name = G.terms_[0].name
    assert G.smoothing_.terms == [f'{name}[1]']
    np.testing.assert_allclose(G.terms_[0].penalty_lams()[0], fixed, rtol=1e-12)
    lam = G.terms_[0].penalty_lams()[1]
    np.testing.assert_allclose(G.smoothing_.lam, [lam], rtol=1e-10)
    # the GAM at those parameters is the same fit
    H = GAM([te('x', 'z', k=5, lam=[fixed, lam]), categorical('c')],
            control=GAMControl(solver='joint', **TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(H.predict(data), G.predict(data), rtol=1e-7, atol=1e-9)
    # estimating both: a smaller criterion, and the first estimate elsewhere
    F = GLMM([te('x', 'z', k=5), categorical('c')], method=method,
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert F.reml_score_ <= G.reml_score_ + 1e-8
    assert abs(np.log(F.smoothing_.lam[0] / fixed)) > 0.1
    # the variance components: both penalties, the fixed one at its value
    vc = G.variance_components_
    np.testing.assert_allclose(vc.loc[f'{name}[0]', 'variance'],
                               G.dispersion_ / (fixed * G.weight_total_), rtol=1e-10)


def test_some_levels_fixed(data):
    # a smooth for each level of a categorical: one level's smoothing
    # parameter fixed
    terms = [categorical('c'), varying(cubic_regression_spline('x', k=6), by='c',
                                       lam=[1e-1, None])]
    G = GLMM(terms, control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.smoothing_.converged
    assert len(G.smoothing_.terms) == 1
    np.testing.assert_allclose(G.terms_[1].penalty_lams()[0], 1e-1, rtol=1e-12)


def test_gam_refuses_none(data):
    with pytest.raises(ValueError, match='GLMM'):
        GAM([te('x', 'z', k=5, lam=[1e-2, None])]).fit(data, data['y'])
