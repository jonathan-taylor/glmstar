"""
Random effects of the combinations of several columns (`re(['batch',
'cask'])`): the same as `re()` of a column of their labels, nested effects
(lme4's `(1 | batch/cask)`), random slopes, new data and positions.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GLMM, GAMControl, s, re

TIGHT = dict(epsnr=1e-12, mxitnr=100)


def _model(method, **kwargs):
    # fixed smoothing is a GAM; estimated, a GLMM
    return GAM(**kwargs) if method == 'fixed' else GLMM(method=method, **kwargs)


@pytest.fixture(scope='module')
def data():
    # 10 batches of 3 casks (labelled a, b, c in each batch: the cask
    # alone does not identify it), 4 observations of each
    rng = np.random.default_rng(5)
    batch = np.repeat(np.arange(10), 12)
    cask = np.tile(np.repeat(list('abc'), 4), 10)
    n = batch.shape[0]
    D = pd.DataFrame({'batch': batch, 'cask': cask, 'x': rng.uniform(0, 1, n),
                      'v': rng.uniform(0, 2, n)})
    b = rng.normal(0, 1, 10)
    c = rng.normal(0, 0.5, 30)
    D['y'] = (np.sin(2 * np.pi * D['x']) + b[batch] + c[batch * 3 + np.searchsorted(list('abc'), cask)]
              + 0.3 * rng.standard_normal(n))
    D['batch_cask'] = D['batch'].astype(str) + ':' + D['cask']
    return D


@pytest.mark.parametrize('method', ['fixed', 'REML', 'ML'])
def test_is_re_of_the_labels(data, method):
    if method != 'fixed':
        pytest.importorskip('jax')
    lam = dict(lam=0.05) if method == 'fixed' else {}
    smooth = s('x', df=4) if method == 'fixed' else s('x')
    control = GAMControl(**TIGHT)
    G = _model(method, terms=[smooth, re('batch', **lam), re(['batch', 'cask'], **lam)],
            control=control).fit(data, data['y'])
    G0 = _model(method, terms=[smooth, re('batch', **lam), re('batch_cask', **lam)],
             control=control).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(data), G0.predict(data), rtol=1e-10)
    effects, effects0 = G.terms_[2].effects_, G0.terms_[2].effects_
    assert list(effects.index) == list(effects0.index)
    assert effects.index[0] == '0:a'
    np.testing.assert_allclose(effects.values, effects0.values, rtol=1e-8, atol=1e-12)
    assert G.terms_[2].name == ('re(batch:cask, lam=0.05)' if method == 'fixed' else 're(batch:cask)')
    if method != 'fixed':
        np.testing.assert_allclose(G.smoothing_.lam, G0.smoothing_.lam, rtol=1e-8)


def test_slopes_and_new_data(data):
    G = GAM(terms=[s('x', df=4), re(['batch', 'cask'], by='v', lam=0.1)]).fit(data, data['y'])
    G0 = GAM(terms=[s('x', df=4), re('batch_cask', by='v', lam=0.1)]).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(data), G0.predict(data), rtol=1e-10)
    assert G.terms_[1].name == 're(batch:cask, by=v, lam=0.1)'
    # a combination not seen in fitting predicts 0 for the term
    new = data.iloc[:2].copy()
    new['cask'] = ['z', 'a']
    new['batch_cask'] = new['batch'].astype(str) + ':' + new['cask']
    f, f0 = G.terms_[1].predict(new[['batch', 'cask', 'v']]), G0.terms_[1].predict(new[['batch_cask', 'v']])
    assert f[0] == 0
    np.testing.assert_allclose(f, f0, rtol=1e-10)


def test_positions(data):
    X = data[['x', 'batch', 'cask']].to_numpy(dtype=object)
    X[:, 0] = X[:, 0].astype(float)
    G = GAM(terms=[s(0, df=4), re(1, lam=0.05), re([1, 2], lam=0.05)]).fit(X, data['y'].values)
    G0 = GAM(terms=[s('x', df=4), re('batch', lam=0.05), re('batch_cask', lam=0.05)]).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(X), G0.predict(data), rtol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_lmer_nested(Rinfo, data, method):
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("lme4", quietly=TRUE))')[0]:
        pytest.skip('requires the R package lme4')
    G = _model(method, terms=[re('batch'), re(['batch', 'cask'])],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data[['batch', 'cask', 'y']])
    rpy.r(f"""D$batch <- factor(D$batch)
              m <- lme4::lmer(y ~ 1 + (1 | batch/cask), data=D, REML={'TRUE' if method == 'REML' else 'FALSE'},
                              control=lme4::lmerControl(optimizer='bobyqa',
                                                        optCtrl=list(rhoend=1e-10)))
              vc <- as.data.frame(lme4::VarCorr(m))""")
    with Rinfo['np_cv_rules'].context():
        sd = dict(zip(rpy.r('vc$grp'), np.asarray(rpy.r('vc$sdcor'))))
        fitted = np.asarray(rpy.r('as.numeric(fitted(m))'))
    ours = G.variance_components_['sd']
    np.testing.assert_allclose(ours['re(batch)'], sd['batch'], rtol=1e-5)
    np.testing.assert_allclose(ours['re(batch:cask)'], sd['cask:batch'], rtol=1e-5)
    np.testing.assert_allclose(ours['scale'], sd['Residual'], rtol=1e-5)
    np.testing.assert_allclose(G.predict(data), fitted, rtol=1e-6)
