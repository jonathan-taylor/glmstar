"""
A modelmatrix Feature (e.g. a log) as the column of a smoother, `s(log_x)`:
the same fits as the smoother of a column computed beforehand.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

pytest.importorskip('scatter_smooth')

from modelmatrix import derived_feature

from glmnet.gam import (GAM, s, lo, linear, varying, random_effect, tensor_spline,
                        categorical, cyclic_spline)

log_x = derived_feature(['x'], encoder=FunctionTransformer(np.log), name='log(x)')
log_z = derived_feature(['z'], encoder=FunctionTransformer(np.log), name='log(z)')


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({'x': rng.uniform(1, 10, n),
                      't': rng.uniform(0, 1, n),
                      'z': rng.uniform(1, 5, n),
                      'g': rng.choice([f'g{i}' for i in range(8)], n)})
    D['y'] = np.sin(2 * np.log(D['x'])) + np.cos(3 * D['t']) + 0.3 * rng.standard_normal(n)
    # the features, computed beforehand
    E = D.assign(logx=np.log(D['x']), logz=np.log(D['z']))
    return D, E


CASES = {
    's': ([s(log_x, df=4), s('t', df=3)], [s('logx', df=4), s('t', df=3)]),
    'lo': ([lo(log_x), s('t', df=3)], [lo('logx'), s('t', df=3)]),
    'linear': ([linear(log_x), s('t', df=3)], [linear('logx'), s('t', df=3)]),
    'cyclic': ([cyclic_spline(log_x, lam=1e-3)], [cyclic_spline('logx', lam=1e-3)]),
    'tensor': ([tensor_spline(log_x, 't', lam=(1e-3, 1e-3))],
               [tensor_spline('logx', 't', lam=(1e-3, 1e-3))]),
    'varying': ([s('t', df=3), varying(s('t', lam=1e-3), by=log_z, as_categorical=False)],
                [s('t', df=3), varying(s('t', lam=1e-3), by='logz', as_categorical=False)]),
    'random-slope': ([s('t', df=3), random_effect('g', by=log_z, lam=1.)],
                     [s('t', df=3), random_effect('g', by='logz', lam=1.)]),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_matches_columns(name):
    D, E = _data()
    terms, ref_terms = CASES[name]
    a = GAM(terms, response_id='y').fit(D, D)
    b = GAM(ref_terms, response_id='y').fit(E, E)
    # new data, the feature computed from its column
    new, new_E = D.iloc[:10], E.iloc[:10]
    np.testing.assert_allclose(a.predict(new), b.predict(new_E), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.edf_.to_numpy(), b.edf_.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(a.partial_effect(0, new), b.partial_effect(0, new_E),
                               rtol=1e-10, atol=1e-12)
    if name != 'lo':   # (loess: no standard errors)
        _, sa = a.partial_effect(0, new, return_std=True)
        _, sb = b.partial_effect(0, new_E, return_std=True)
        np.testing.assert_allclose(sa, sb, rtol=1e-10, atol=1e-12)


def test_names():
    D, _ = _data()
    G = GAM([s(log_x, df=4), random_effect('g', by=log_z, lam=1.),
             varying(s('t', lam=1e-3), by=log_z, as_categorical=False)], response_id='y')
    assert list(G.fit(D, D).edf_.index) == [
        's(log(x), df=4)', 'random_effect(g, by=log(z), lam=1.0)',
        'varying(s(t, lam=0.001), by=log(z), as_categorical=False)']
    tex = G.model_latex()
    assert r'\text{log(x)}_i' in tex and r'\text{log(z)}' in tex
    axes = G.plot(terms=[0])
    assert axes[0].get_xlabel() == 'log(x)'


def test_errors():
    D, _ = _data()
    with pytest.raises(ValueError, match='grouping'):
        GAM([random_effect(log_x)], response_id='y').fit(D, D)
    with pytest.raises(ValueError, match='grouping'):
        GAM([categorical(log_x)], response_id='y').fit(D, D)
    with pytest.raises(ValueError, match='grouping'):
        GAM([varying(s('t', lam=1e-3), by=log_z)], response_id='y').fit(D, D)
    other = derived_feature(['z'], encoder=FunctionTransformer(np.log), name='log(x)')
    with pytest.raises(ValueError, match='two different features'):
        GAM([s(log_x, df=4), linear(other)], response_id='y').fit(D, D)
    two = derived_feature(['x', 'z'], name='xz')
    with pytest.raises(ValueError, match='should have one'):
        GAM([s(two, df=4)], response_id='y').fit(D, D)
    # the smooth of a feature and the feature as a fixed effect: not identified
    with pytest.raises(ValueError, match='not identified'):
        GAM([s(log_x, df=4), log_x], response_id='y').fit(D, D)
    # the column itself, linear, and the smooth of its log: identified
    GAM([s(log_x, df=4), 'x'], response_id='y').fit(D, D)
