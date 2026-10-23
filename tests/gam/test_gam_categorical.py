"""
`categorical()` of a pandas categorical keeps its categories' order (those
seen in fitting): the effects, the plots and the baseline of modified
backfitting follow it, and the fit is the same as with the sorted levels.
`factor()`, its former name, is deprecated.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, s, categorical

MONTHS = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    month = rng.choice(MONTHS[:11], n)          # December not observed
    D = pd.DataFrame({'month': pd.Categorical(month, categories=MONTHS),
                      'x': rng.uniform(0, 1, n)})
    effect = dict(zip(MONTHS, np.sin(np.arange(12) / 2)))
    D['y'] = np.sin(2 * np.pi * D['x']) + D['month'].map(effect).astype(float) + \
        0.3 * rng.standard_normal(n)
    return D


@pytest.mark.parametrize('solver', ['backfit', 'joint'])
def test_categorical_order(data, solver):
    control = GAMControl(solver=solver, epsnr=1e-12, bf_tol=1e-13, bf_maxit=5000)
    G = GAM(terms=[s('x', df=4), categorical('month')], control=control).fit(data, data['y'])
    effects = G.terms_[1].effects_
    assert list(effects.index) == MONTHS[:11]
    # the same fit as with the sorted labels (strings)
    D = data.assign(month=data['month'].astype(str))
    G0 = GAM(terms=[s('x', df=4), categorical('month')], control=control).fit(D, D['y'])
    assert list(G0.terms_[1].effects_.index) == sorted(MONTHS[:11])
    np.testing.assert_allclose(G.predict(data), G0.predict(D), rtol=1e-8)
    np.testing.assert_allclose(effects.values, G0.terms_[1].effects_.loc[MONTHS[:11]].values,
                               rtol=1e-8, atol=1e-10)
    # a month not seen in fitting (December) predicts 0 for the term
    new = data.iloc[:3].copy()
    new['month'] = pd.Categorical(['Dec', 'Jan', 'Dec'], categories=MONTHS)
    f = G.terms_[1].predict(new['month'])
    assert f[0] == 0 and f[2] == 0 and f[1] == effects['Jan']
    np.testing.assert_allclose(G.terms_[1].basis(new['month'])[[0, 2]], 0)


def test_plain_columns_sorted(data):
    D = data.assign(month=data['month'].astype(str))
    G = GAM(terms=[categorical('month')]).fit(D, D['y'])
    assert list(G.terms_[0].effects_.index) == sorted(MONTHS[:11])


def test_factor_is_deprecated():
    # factor() and FactorTerm: categorical() and CategoricalSmoother, with a
    # FutureWarning; the term keeps its former name
    from glmnet.gam import factor, CategoricalSmoother
    with pytest.warns(FutureWarning, match='use categorical'):
        term = factor('c')
    assert isinstance(term, CategoricalSmoother) and term.name == 'factor(c)'
    with pytest.warns(FutureWarning, match='FactorTerm is deprecated.*use CategoricalSmoother'):
        from glmnet.gam import FactorTerm
    old = FactorTerm('c')
    assert isinstance(old, CategoricalSmoother) and old.name == 'factor(c)'
    D = pd.DataFrame({'c': list('xyzxyzxy'), 'y': np.arange(8.)})
    with pytest.warns(FutureWarning):
        G = GAM(terms=[factor('c')]).fit(D, D['y'])
    H = GAM(terms=[categorical('c')]).fit(D, D['y'])
    np.testing.assert_allclose(G.predict(D), H.predict(D))
    assert list(H.edf_.index) == ['categorical(c)']
