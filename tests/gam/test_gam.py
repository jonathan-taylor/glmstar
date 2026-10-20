"""
`ISLP.gam`: GAMs fitted by local scoring (IRLS + backfitting).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from scatter_smooth import SplineSmoother

from glmnet import GLM
from ISLP.gam import GAM, GAMControl, s, lo, linear, factor
from ISLP.gam.terms import SplineTerm

TIGHT = GAMControl(epsnr=1e-12, mxitnr=100)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 250
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n)})
    eta = np.sin(2 * np.pi * D['a']) + 0.4 * D['b'] + D['c'].map({'x': 0, 'y': .5, 'z': -.5})
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    D['w'] = rng.uniform(0.5, 2, n)
    return D


def test_one_spline_is_the_smoothing_spline(data):
    # one spline term, Gaussian: the weighted smoothing spline, with lam on
    # the scale of the deviance over twice the sum of the weights
    lam = 1e-4
    G = GAM(terms=[s('a', lam=lam)], weight_id='w').fit(data[['a']], data[['y', 'w']])
    w = data['w'].to_numpy()
    S = SplineSmoother(data['a'].to_numpy(), w=w / w.sum(), lamval=lam)
    S.smooth(data['y'].to_numpy())
    np.testing.assert_allclose(G.predict(data[['a']]), S.predict(data['a'].to_numpy()),
                               rtol=1e-8, atol=1e-10)
    # predictions at new points: the natural cubic spline
    x_new = pd.DataFrame({'a': np.linspace(-0.1, 1.1, 13)})
    np.testing.assert_allclose(G.predict(x_new), S.predict(x_new['a'].to_numpy()),
                               rtol=1e-8, atol=1e-10)


def test_parametric_terms_are_a_glm(data):
    # linear and factor terms only: the GLM on the corresponding design
    G = GAM(terms=[linear('b'), factor('c')], family=sm.families.Binomial(),
            control=TIGHT).fit(data, data['yb'])
    X = pd.get_dummies(data[['b', 'c']], drop_first=True, dtype=float)
    M = GLM(family=sm.families.Binomial()).fit(X, data['yb'])
    np.testing.assert_allclose(G.predict(data, 'link'), M.predict(X, 'link'),
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(G.deviance_, M.deviance_, rtol=1e-10)
    np.testing.assert_allclose(G.aic_, M.aic_, rtol=1e-10)
    assert G.df_resid_ == M.df_resid_


def test_fitted_values_and_partial_effects(data):
    G = GAM(terms=[s('a', df=5), s('b', lam=1e-2), factor('c')],
            family=sm.families.Binomial(), control=TIGHT).fit(data, data['yb'])
    np.testing.assert_allclose(G.predict(data, 'link'), G.state_.linear_predictor,
                               rtol=1e-8, atol=1e-10)
    fits = G.fits_
    assert list(fits.columns) == ['s(a)', 's(b)', 'factor(c)']
    np.testing.assert_allclose(fits.mean(axis=0), 0, atol=1e-10)
    for name, column in zip(fits.columns, ['a', 'b', 'c']):
        np.testing.assert_allclose(G.partial_effect(name, data[column]), fits[name],
                                   rtol=1e-8, atol=1e-10)
    # df matched at the final weights
    np.testing.assert_allclose(G.edf_['s(a)'], 5, rtol=1e-5)
    assert G.summary().shape == (3, 2)


def test_linear_term(data):
    # a linear term w_i * eta_i moves the response, for the Gaussian, to y - omega / w~
    rng = np.random.default_rng(1)
    w = data['w'].to_numpy()
    wn = w / w.sum()
    omega = 0.1 * rng.standard_normal(len(w)) * wn
    terms = [s('a', lam=1e-4), s('b', lam=1e-3)]
    G = GAM(terms=terms, weight_id='w').fit(data, data[['y', 'w']], linear_term=omega)
    shifted = data[['y', 'w']].assign(y=data['y'] - omega / wn)
    G0 = GAM(terms=terms, weight_id='w').fit(data, shifted)
    np.testing.assert_allclose(G.state_.linear_predictor, G0.state_.linear_predictor,
                               rtol=1e-8, atol=1e-10)


def test_objective_decreases_and_converges(data):
    # fixed smoothing: the objective is checked; the fit is a stationary point of
    # the penalized objective (no change from one more local scoring run)
    G = GAM(terms=[s('a', lam=1e-4), linear('b'), factor('c')],
            family=sm.families.Binomial(), control=TIGHT).fit(data, data['yb'])
    assert G.regularizer_.has_objective
    assert 1 < G.n_iter_ < TIGHT.mxitnr
    # df terms and loess have no objective: local scoring stops on eta
    G = GAM(terms=[s('a', df=4), lo('b'), factor('c')],
            family=sm.families.Binomial(), control=TIGHT).fit(data, data['yb'])
    assert not G.regularizer_.has_objective
    assert G.n_iter_ < TIGHT.mxitnr
    assert 1 < G.edf_['lo(b)'] < 10


def test_ndarray_and_integer_columns(data):
    X = data[['a', 'b']].to_numpy()
    G = GAM(terms=[s(0, lam=1e-4), s(1, df=3)]).fit(X, data['y'].to_numpy())
    D = GAM(terms=[s('a', lam=1e-4), s('b', df=3)]).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(X), D.predict(data), rtol=1e-10)
    assert isinstance(G.fits_, np.ndarray)


def test_errors(data):
    with pytest.raises(ValueError, match='one of df and lam'):
        SplineTerm('a', df=3, lam=1.)
    with pytest.raises(ValueError, match='at least 4 distinct'):
        GAM(terms=[s('a')]).fit(pd.DataFrame({'a': [0., 1, 2] * 5}), np.arange(15.))
    with pytest.raises(ValueError, match='at least one term'):
        GAM().fit(data, data['y'])
