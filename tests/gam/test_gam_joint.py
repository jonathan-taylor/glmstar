"""
The joint (mgcv-style) solve of the working problems against backfitting,
the terms' design operators, random effects and the Bayesian covariance.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from ISLP.models import ModelSpec as MS, ns, contrast
from ISLP.gam import GAM, GAMControl, s, lo, re, linear, factor

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)
GROUPS = [f'g{i:02d}' for i in range(12)]


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 400
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n),
                      'g': rng.choice(GROUPS, n),
                      'v': rng.uniform(0, 2, n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    effects = dict(zip(GROUPS, rng.normal(0, 0.7, len(GROUPS))))
    eta = (np.sin(2 * np.pi * D['a']) + 0.4 * D['b'] +
           D['c'].map({'x': 0, 'y': .5, 'z': -.5}) + D['g'].map(effects))
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(eta + D['o']))))
    D['yp'] = rng.poisson(np.exp(0.3 * eta + D['o']))
    return D


FAMILIES = {'gaussian': (sm.families.Gaussian, 'y'),
            'binomial': (sm.families.Binomial, 'yb'),
            'poisson': (sm.families.Poisson, 'yp')}
TERMS = {
    'splines and a factor': lambda: [s('a', lam=1e-3), s('b', df=3), factor('c')],
    'linear and a random intercept': lambda: [s('a', df=4), linear('b'), re('g', lam=0.05)],
    'random intercept and slope': lambda: [s('a', df=4), re('g', df=6), re('g', by='v', lam=0.1)],
}


@pytest.mark.parametrize('terms', list(TERMS))
@pytest.mark.parametrize('family', list(FAMILIES))
def test_joint_is_backfit(data, family, terms):
    # the same working problems, solved jointly or by (plain or modified)
    # backfitting: the same fit, with prior weights and an offset
    klass, y = FAMILIES[family]
    kw = dict(response_id=y, weight_id='w', offset_id='o')
    fits = {(solver, modified): GAM(terms=TERMS[terms](), family=klass(),
                                    control=GAMControl(**TIGHT, solver=solver, bf_modified=modified),
                                    **kw).fit(data, data[[y, 'w', 'o']])
            for solver, modified in [('joint', True), ('backfit', True), ('backfit', False)]}
    J = fits['joint', True]
    for key, G in fits.items():
        np.testing.assert_allclose(G.state_.link_parameter, J.state_.link_parameter,
                                   rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(G.edf_, J.edf_, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(G.edf_influence_, J.edf_influence_, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(G.posterior_covariance_, J.posterior_covariance_, rtol=1e-6, atol=1e-10)
    # predictions at the training data are the fit (offset added back)
    np.testing.assert_allclose(J.predict(data, 'link', offset=data['o']),
                               J.state_.link_parameter, rtol=1e-10, atol=1e-10)


def test_joint_model_spec(data):
    # parametric blocks of a ModelSpec (a basis, a contrast) and smooths
    spec = MS([s('a', df=4), ns('b', df=3), contrast('c', method='sum'), re('g', lam=0.05)])
    fits = [GAM(model_spec=spec, control=GAMControl(**TIGHT, solver=solver)).fit(data, data['y'])
            for solver in ('backfit', 'joint')]
    np.testing.assert_allclose(fits[0].state_.linear_predictor, fits[1].state_.linear_predictor,
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fits[1].predict(data), fits[1].state_.linear_predictor,
                               rtol=1e-10, atol=1e-10)
    # the random effects are named by the groups
    effects = fits[1].terms_[3].effects_
    assert list(effects.index) == GROUPS


@pytest.mark.parametrize('x', ['coarse', 'fine'])
def test_spline_operators(x):
    # the basis reproduces the fit from its coefficients, and the penalty
    # matrix gives the roughness (Reinsch engine: knots at the distinct
    # values; B-spline engine: R's thinned knots)
    rng = np.random.default_rng(1)
    v = np.round(rng.uniform(0, 3, 300), 1) if x == 'coarse' else rng.uniform(0, 3, 600)
    D = pd.DataFrame({'x': v, 'y': np.sin(2 * v) + 0.3 * rng.standard_normal(v.shape[0])})
    G = GAM(terms=[s('x', lam=1e-3)], control=GAMControl(**TIGHT)).fit(D, D['y'])
    term = G.terms_[0]
    coef = G.state_.coefs[0]
    B = term.basis()
    np.testing.assert_allclose(B @ coef, G.fits_.iloc[:, 0], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(B.sum(1), 1, atol=1e-12)       # a partition of unity
    Omega = term.penalty_matrix()
    np.testing.assert_allclose(coef @ Omega @ coef, term.roughness(coef), rtol=1e-9)
    new = np.linspace(-1, 4, 41)              # beyond the knots: linear
    np.testing.assert_allclose(term.basis(new) @ coef, term.predict(new), rtol=1e-8, atol=1e-10)
    # the basis (from its values at the knots) is the smoother's splines
    # whose coefficients are the unit vectors
    np.testing.assert_allclose(term.basis(new), term._unit_values(new), atol=1e-12)
    np.testing.assert_allclose(B, term._unit_values(term.x_), atol=1e-12)


def test_random_effect_term(data):
    G = GAM(terms=[re('g', lam=0.02)], control=GAMControl(**TIGHT)).fit(data, data['y'])
    term = G.terms_[0]
    # a ridge of the group means: b_g = sum(w (y - intercept)) / (sum(w) + lam),
    # at the normalized weights (1 / n each)
    n = data.shape[0]
    r = data['y'] - G.intercept_
    sums = r.groupby(data['g']).sum() / n
    counts = data['g'].value_counts().sort_index() / n
    np.testing.assert_allclose(term.effects_.loc[GROUPS], (sums / (counts + 0.02)).loc[GROUPS],
                               rtol=1e-8)
    # not centered: the intercept and the effects as fitted
    np.testing.assert_allclose(G.predict(data), G.intercept_ + term.effects_.loc[data['g']].values,
                               rtol=1e-10)
    # unseen groups predict 0, edf the sum of the shrinkage factors
    new = pd.DataFrame({'g': ['g00', 'new']})
    assert G.partial_effect(0, new['g'])[1] == 0
    np.testing.assert_allclose(G.edf_.iloc[0], (counts / (counts + 0.02)).sum(), rtol=1e-10)
    # df: the shrinkage matched to it
    G = GAM(terms=[re('g', df=5)], control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(G.edf_.iloc[0], 5, rtol=1e-8)
    with pytest.raises(ValueError, match='give lam or df'):
        GAM(terms=[re('g')]).fit(data, data['y'])


def test_parametric_covariance_is_least_squares(data):
    # no penalties: Vp is the least squares covariance, the standard errors
    # those of the OLS fitted values (at the GAM's dispersion)
    G = GAM(terms=[linear('a'), linear('b'), factor('c')],
            control=GAMControl(**TIGHT, solver='joint')).fit(data, data['y'])
    X = sm.add_constant(pd.get_dummies(data[['a', 'b', 'c']], drop_first=True).astype(float))
    ols = sm.OLS(data['y'], X).fit()
    np.testing.assert_allclose(G.dispersion_, ols.scale, rtol=1e-10)
    pred, se = G.predict(data.iloc[:20], 'link', return_std=True)
    sf = ols.get_prediction(X.iloc[:20]).se_mean
    np.testing.assert_allclose(se, sf, rtol=1e-8)
    np.testing.assert_allclose(G.edf_influence_.values, [1, 1, 2], rtol=1e-10)


def test_standard_errors_and_errors(data):
    G = GAM(terms=[s('a', df=4), factor('c')], family=sm.families.Binomial(),
            control=GAMControl(**TIGHT)).fit(data, data['yb'])
    mu, se = G.predict(data.iloc[:5], return_std=True)
    eta, se_eta = G.predict(data.iloc[:5], 'link', return_std=True)
    np.testing.assert_allclose(se, se_eta * mu * (1 - mu), rtol=1e-10)
    fit, se = G.partial_effect('s(a, df=4)', data['a'].iloc[:5], return_std=True)
    assert np.all(se > 0)
    # loess: no basis, no joint solve or standard errors
    with pytest.raises(ValueError, match='has no basis'):
        GAM(terms=[lo('a')], control=GAMControl(solver='joint')).fit(data, data['y'])
    G = GAM(terms=[lo('a'), s('b', df=3)]).fit(data, data['y'])
    assert G.posterior_covariance_ is None and G.edf_influence_ is None
    with pytest.raises(ValueError, match='standard errors need'):
        G.predict(data, return_std=True)
    with pytest.raises(ValueError, match="solver should be"):
        GAM(terms=[s('a')], control=GAMControl(solver='qr')).fit(data, data['y'])


def test_covariance_when_used(data):
    # the covariance is computed when first used, then kept
    G = GAM(terms=[s('a', df=4), s('b', lam=0.1), factor('c')],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G._covariance is None
    V = G.posterior_covariance_
    assert G._covariance is not None and G.posterior_covariance_ is V
    assert G.frequentist_covariance_.shape == V.shape == (G.design_.X.shape[1],) * 2
    # the same as computed from the start, by a refit
    H = GAM(terms=[s('a', df=4), s('b', lam=0.1), factor('c')],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    _, std = H.predict(data.iloc[:10], return_std=True)
    np.testing.assert_allclose(std, G.predict(data.iloc[:10], return_std=True)[1], rtol=1e-12)
    np.testing.assert_allclose(H.edf_influence_, G.edf_influence_, rtol=1e-12)
    assert list(G.summary().columns) == ['edf', 'edf_influence', 'lam']
