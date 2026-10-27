"""
One coding of the levels of a grouping: a pandas categorical's order (its
categories seen in fitting), otherwise sorted, as modelmatrix orders a
categorical's levels; for `categorical`, a categorical `varying`, and now
the random effects too. The fits do not depend on the order.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import (GAM, GLMM, s, categorical, random_effect, varying,
                        correlated_random_effect)
from glmnet.gam.reml import ExperimentalWarning

ORDER = ['g3', 'g0', 'g5', 'g1', 'g4', 'g2', 'g6']   # not sorted; g6 never seen


def _data(seed=0, n=240):
    rng = np.random.default_rng(seed)
    g = rng.choice(ORDER[:-1], n)
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.standard_normal(n), 'g': g})
    effect = dict(zip(ORDER, rng.standard_normal(len(ORDER))))
    D['y'] = (np.sin(4 * D['x']) + D['g'].map(effect) + 0.5 * D['z'] * (D['g'] == 'g1')
              + 0.3 * rng.standard_normal(n))
    C = D.assign(g=pd.Categorical(D['g'], categories=ORDER))
    return D, C


TERMS = {
    'random_effect': [s('x', df=4), random_effect('g', lam=0.5)],
    'random_slope': [s('x', df=4), random_effect('g', by='z', lam=0.5)],
    'correlated': [s('x', df=4), correlated_random_effect('g', slopes=['z'],
                                                          precision=np.eye(2))],
    'categorical': [s('x', df=4), categorical('g')],
    'varying': [categorical('g'), varying(s('x', lam=1e-3), by='g')],
}


@pytest.mark.parametrize('name', sorted(TERMS))
def test_order_of_a_categorical(name):
    D, C = _data()
    a = GAM(TERMS[name], response_id='y').fit(C, C)    # the categorical's order
    b = GAM(TERMS[name], response_id='y').fit(D, D)    # sorted
    term = a.terms_[1]
    assert list(term.levels_) == [g for g in ORDER if g in set(D['g'])]
    assert list(b.terms_[1].levels_) == sorted(set(D['g']))
    # (the same fit, to the iterations' convergence)
    np.testing.assert_allclose(a.predict(C), b.predict(D), rtol=1e-7, atol=1e-9)
    # a level not seen in fitting predicts 0 (the categorical's level g6 too)
    new = pd.DataFrame({'x': [0.5, 0.5], 'z': [1., 1.],
                        'g': pd.Categorical(['g6', 'g1'], categories=ORDER)})
    plain = new.assign(g=new['g'].astype(str))
    np.testing.assert_allclose(a.partial_effect(1, new), b.partial_effect(1, plain),
                               rtol=1e-7, atol=1e-9)
    assert a.partial_effect(1, new)[0] == 0


def test_glmm_order():
    pytest.importorskip('jax')
    D, C = _data()
    terms = [s('x'), random_effect('g')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        a = GLMM(terms, response_id='y').fit(C, C)
        b = GLMM(terms, response_id='y').fit(D, D)
    np.testing.assert_allclose(a.smoothing_.lam, b.smoothing_.lam, rtol=1e-6)
    np.testing.assert_allclose(a.predict(C), b.predict(D), rtol=1e-6, atol=1e-8)
    effects = a.terms_[1].effects_
    assert list(effects.index) == [g for g in ORDER if g in set(D['g'])]
    np.testing.assert_allclose(effects.loc[sorted(effects.index)].to_numpy(),
                               b.terms_[1].effects_.to_numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a.variance_components_.to_numpy(float),
                               b.variance_components_.to_numpy(float), rtol=1e-6)


def test_interaction_is_sorted():
    # the combinations of an interaction's columns: sorted, as before
    D, C = _data()
    C['h'] = pd.Categorical(np.where(C['z'] > 0, 'b', 'a'), categories=['b', 'a'])
    G = GAM([s('x', df=4), random_effect(['g', 'h'], lam=0.5)], response_id='y').fit(C, C)
    levels = list(G.terms_[1].levels_)
    assert levels == sorted(levels)
