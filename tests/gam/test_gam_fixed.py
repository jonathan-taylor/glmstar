"""
Fixed effects of a GAM from modelmatrix (`glmnet.gam.fixed`): the items of
`terms` other than smoothers (columns, interactions, modelmatrix features,
a wildcard, a ModelSpec) are blocks of columns of a ModelSpec. Checked
against the same GAM with the blocks built by hand, statsmodels for a
parametric model and mgcv for a smooth with a factor, a polynomial and an
interaction.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, KFold

pytest.importorskip('scatter_smooth')

from modelmatrix import ModelSpec as MS, Outcome, contrast, poly, wildcard

from glmnet.cox import CoxFamily
from glmnet.gam import GAM, GLMM, GAMControl, s, linear, random_effect, varying
from glmnet.gam.terms import ColumnsSmoother
from glmnet.gam.reml import ExperimentalWarning

POISSON = sm.families.Poisson()


def _data(seed=0, n=240):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({'x': rng.uniform(0, 1, n),
                      'z': rng.standard_normal(n),
                      'w': rng.standard_normal(n),
                      'c': pd.Categorical(rng.choice(list('abc'), n)),
                      'g': rng.choice([f'g{i}' for i in range(10)], n),
                      'f': pd.Categorical(rng.choice(list('pq'), n))})
    eta = np.sin(6 * D['x']) + D['z'] + 0.5 * D['w'] ** 2 + 0.3 * D['c'].cat.codes
    D['y'] = eta + rng.standard_normal(n)
    D['cnt'] = rng.poisson(np.exp(0.3 * eta))
    D['time'] = rng.exponential(1 / np.exp(0.3 * eta)) + 0.01
    D['status'] = rng.binomial(1, 0.7, n)
    return D


ITEMS = ['z', contrast('c'), poly('w', 2), ('z', 'c')]
NAMES = ['z', 'c', 'poly(w, degree=2)', 'z:c']


def _by_hand(D, items=ITEMS):
    # the blocks of a ModelSpec's columns, built by hand
    spec = MS(items).fit(D)
    Z = spec.transform(D)
    E = pd.concat([D, Z.add_prefix('m:')], axis=1)
    blocks = [ColumnsSmoother([f'm:{c}' for c in spec.column_names_[t]], label=f.name)
              for t, f in zip(spec.expanded_terms_, spec.terms_)]
    return blocks, E


def _same(a, Da, b, Db, cox=False):
    np.testing.assert_allclose(a.predict(Da, prediction_type='link'),
                               b.predict(Db, prediction_type='link'), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.edf_.to_numpy(), b.edf_.to_numpy(), rtol=1e-10)
    assert list(a.edf_.index) == list(b.edf_.index)
    fit_a, se_a = a.partial_effect('poly(w, degree=2)', Da.iloc[:20], return_std=True)
    fit_b, se_b = b.partial_effect('poly(w, degree=2)', Db.iloc[:20], return_std=True)
    np.testing.assert_allclose(fit_a, fit_b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(se_a, se_b, rtol=1e-10, atol=1e-12)
    if not cox:
        fa, fb = a.fixed_effects_, b.fixed_effects_
        np.testing.assert_allclose(fa.to_numpy(), fb.to_numpy(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a.fixed_effects_covariance_.to_numpy(),
                                   b.fixed_effects_covariance_.to_numpy(), rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize('family', ['gaussian', 'poisson', 'cox'])
def test_matches_blocks(family):
    D = _data()
    kw = {'gaussian': dict(response_id='y'),
          'poisson': dict(family=POISSON, response_id='cnt'),
          'cox': dict(family=CoxFamily(event_id='time', status_id='status'))}[family]
    a = GAM([s('x', df=4)] + ITEMS, **kw).fit(D, D)
    b = GAM([s('x', df=4), MS(ITEMS)], **kw).fit(D, D)
    blocks, E = _by_hand(D)
    c = GAM([s('x', df=4)] + blocks, **kw).fit(E, E)
    assert list(a.edf_.index) == ['s(x, df=4)'] + NAMES
    _same(a, D, b, D, cox=family == 'cox')
    _same(a, D, c, E, cox=family == 'cox')
    if family != 'cox':
        assert list(a.fixed_effects_.index) == [
            'intercept', 'z', 'c[b]', 'c[c]', 'poly(w, degree=2)[0]', 'poly(w, degree=2)[1]',
            'z:c[b]', 'z:c[c]']


def test_glmm_matches_blocks():
    pytest.importorskip('jax')
    D = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        a = GLMM([s('x'), random_effect('g')] + ITEMS, family=POISSON,
                 response_id='cnt').fit(D, D)
        blocks, E = _by_hand(D)
        c = GLMM([s('x'), random_effect('g')] + blocks, family=POISSON,
                 response_id='cnt').fit(E, E)
    np.testing.assert_allclose(a.smoothing_.lam, c.smoothing_.lam, rtol=1e-8)
    np.testing.assert_allclose(a.predict(D), c.predict(E), rtol=1e-8)
    np.testing.assert_allclose(a.fixed_effects_.to_numpy(), c.fixed_effects_.to_numpy(),
                               rtol=1e-7, atol=1e-10)


def test_parametric_matches_statsmodels():
    # only fixed effects: a GLM
    D = _data()
    G = GAM(ITEMS, family=POISSON, response_id='cnt',
            control=GAMControl(epsnr=1e-14, mxitnr=100)).fit(D, D)
    X = MS(ITEMS, intercept=True).fit_transform(D)
    ref = sm.GLM(D['cnt'], X, family=POISSON).fit(tol=1e-14)
    np.testing.assert_allclose(G.predict(D, prediction_type='link'), ref.predict(X, which='linear'),
                               rtol=1e-8)
    # (the GAM's intercept is at the columns' means)
    fe = G.fixed_effects_.iloc[1:]
    np.testing.assert_allclose(fe['estimate'].to_numpy(), ref.params.to_numpy()[1:], rtol=1e-7)
    np.testing.assert_allclose(fe['std_error'].to_numpy(), ref.bse.to_numpy()[1:], rtol=1e-6)


def test_order_and_wildcard():
    D = _data()
    G = GAM([s('x', df=4), 'z', s('w', df=3), contrast('c')], response_id='y').fit(D, D)
    assert list(G.edf_.index) == ['s(x, df=4)', 'z', 's(w, df=3)', 'c']
    # a wildcard leaves out the terms given, and the outcome's columns
    outcome = Outcome('y')
    G = GAM([s('x', df=4), 'z', wildcard(exclude=[outcome, 'x', 'g', 'f', 'cnt', 'time',
                                                    'status'])],
            outcome=outcome).fit(D, D)
    assert list(G.edf_.index) == ['s(x, df=4)', 'z', 'w', 'c']


def test_identifiability_guard():
    D = _data()
    with pytest.raises(ValueError, match="not identified.*'x'"):
        GAM([s('x', df=4), 'x'], response_id='y').fit(D, D)
    with pytest.raises(ValueError, match='not identified'):
        GAM([s('x', df=4), poly('x', 2)], response_id='y').fit(D, D)
    with pytest.raises(ValueError, match='not identified'):
        GAM([linear('z'), MS(['z'])], response_id='y').fit(D, D)
    # a factor of a varying term, an interaction: identified
    GAM([varying(s('x', lam=1e-3), by='f'), contrast('f'), ('z', 'f')], response_id='y').fit(D, D)


def test_errors():
    D = _data()
    with pytest.raises(ValueError, match='no intercept'):
        GAM([s('x', df=4), MS(['z'], intercept=True)], response_id='y').fit(D, D)
    G = GAM([s('x', df=4), contrast('c')], response_id='y').fit(D, D)
    new = D.iloc[:5].assign(c=pd.Categorical(['d'] * 5))
    with pytest.raises(ValueError, match='unknown categories'):
        G.predict(new)


def test_latex():
    G = GAM([s('x', df=4), 'z', poly('w', 2), ('z', 'c'), MS([contrast('c', 'sum')]),
             wildcard(exclude=['y'])])
    tex = G.model_latex()
    for name in ['z', 'poly(w, degree=2)', 'z:c', 'c', 'the other columns']:
        assert rf'\text{{{name}}}' in tex
    D = _data()
    G = GAM([s('x', df=4), 'z'], response_id='y').fit(D, D)
    assert G.model_latex() == GAM([s('x', df=4), 'z']).model_latex()


def test_grid_search():
    D = _data()
    G = GAM([s('x', df=4), MS(['z', poly('w', 2)])], response_id='y')
    assert G.get_params()['terms__1__terms__1__degree'] == 2
    search = GridSearchCV(G, {'terms__1__terms__1__degree': [1, 2, 3]},
                          cv=KFold(3, shuffle=True, random_state=0)).fit(D, D)
    assert search.best_params_['terms__1__terms__1__degree'] in (2, 3)
    # a plain item is replaced whole
    G = GAM([s('x', df=4), 'z', poly('w', 2)], response_id='y')
    G.set_params(terms__2=poly('w', 3))
    assert G.fit(D, D).edf_['poly(w, degree=3)'] == 3


def test_polars():
    pl = pytest.importorskip('polars')
    D = _data()
    a = GAM([s('x', df=4)] + ITEMS, response_id='y').fit(D, D)
    P = pl.from_pandas(D)
    b = GAM([s('x', df=4)] + ITEMS, response_id='y').fit(P, P)
    np.testing.assert_allclose(a.predict(D), b.predict(P), rtol=1e-10)


def test_mgcv(Rinfo):
    # mgcv's gam(y ~ s(x, bs='cr') + c + z + poly(w, 2) + z:c, sp=...): the same
    # linear predictor and parametric coefficients
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    D = _data()
    D['x'] = np.round(D['x'] * 40) / 40   # 41 distinct values: the knots
    lam = 1e-4
    G = GAM([s('x', lam=lam), contrast('c'), 'z', poly('w', 2), ('z', 'c')], response_id='y',
            control=GAMControl(epsnr=1e-12, mxitnr=100)).fit(D, D)
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D[['x', 'z', 'w', 'y']].assign(c=D['c'].astype(str)))
    k = D['x'].nunique()

    def mgcv(sp):
        rpy.r(f"""D$c <- factor(D$c)
                  m <- mgcv::gam(y ~ s(x, bs='cr', k={k}) + c + z + poly(w, 2) + z:c, data=D,
                                 knots=list(x=sort(unique(D$x))), sp={float(sp)!r})""")
    mgcv(1.)
    scale = rpy.r('m$smooth[[1]]$S.scale')[0]
    mgcv(lam * D.shape[0] * scale)
    np.testing.assert_allclose(G.state_.link_parameter, np.asarray(rpy.r('m$linear.predictors')),
                               rtol=1e-7, atol=1e-8)
    coef = dict(zip(rpy.r('names(coef(m))'), np.asarray(rpy.r('coef(m)'))))
    fe = G.fixed_effects_['estimate']
    for ours, R in [('c[b]', 'cb'), ('c[c]', 'cc'), ('z', 'z'), ('z:c[b]', 'cb:z'),
                    ('z:c[c]', 'cc:z'), ('poly(w, degree=2)[0]', 'poly(w, 2)1'),
                    ('poly(w, degree=2)[1]', 'poly(w, 2)2')]:
        np.testing.assert_allclose(fe[ours], coef[R], rtol=1e-6, atol=1e-8)
