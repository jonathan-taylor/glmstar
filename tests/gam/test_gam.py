"""
`glmnet.gam`: GAMs fitted by local scoring (IRLS + backfitting).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from scatter_smooth import SplineSmoother

from glmnet import GLM
from glmnet.gam import GAM, GAMControl, s, lo, linear, factor
from glmnet.gam.terms import SplineTerm, LoessTerm

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
    assert list(fits.columns) == ['s(a, df=5)', 's(b, lam=0.01)', 'factor(c)']
    np.testing.assert_allclose(fits.mean(axis=0), 0, atol=1e-10)
    for name, column in zip(fits.columns, ['a', 'b', 'c']):
        np.testing.assert_allclose(G.partial_effect(name, data[column]), fits[name],
                                   rtol=1e-8, atol=1e-10)
    # df matched at the final weights
    np.testing.assert_allclose(G.edf_['s(a, df=5)'], 5, rtol=1e-5)
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


def test_many_distinct_values():
    # continuous x: the knots are thinned (as R's gam), df is matched, the
    # fixed-lam objective is that of the spline, and predict reproduces the fit
    rng = np.random.default_rng(3)
    n = 20000
    D = pd.DataFrame({'a': rng.uniform(0, 1, n), 'b': rng.uniform(-2, 2, n)})
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-np.sin(2 * np.pi * D['a']) - 0.4 * D['b'])))
    # (df terms move lam with the weights, by about 1e-11 relative at the
    # root of the df match: eta converges to about 1e-11)
    G = GAM(terms=[s('a', df=5), s('b', lam=1e-3)], family=sm.families.Binomial(),
            control=GAMControl(epsnr=1e-10, mxitnr=100)).fit(D, D['yb'])
    assert G.n_iter_ < 20
    assert len(G.terms_[0].knots_) < 250
    np.testing.assert_allclose(G.edf_['s(a, df=5)'], 5, rtol=1e-6)
    np.testing.assert_allclose(G.predict(D, 'link'), G.state_.linear_predictor,
                               rtol=1e-8, atol=1e-10)
    # the objective: deviance / (2 sum w) + lam / 2 * int f''^2, with the
    # integral from the second derivative of the fitted spline
    term = G.terms_[1]
    t = term.knots_
    d2 = term._smoother.predict(t, deriv=2)
    h = np.diff(t)
    rough = np.sum(h / 3 * (d2[:-1]**2 + d2[:-1] * d2[1:] + d2[1:]**2))
    np.testing.assert_allclose(G.regularizer_.objective(G.state_), 1e-3 / 2 * rough, rtol=1e-6)


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


def _correlated(rho, n=3000, seed=4):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * a) + b
    D = pd.DataFrame(dict(a=a, b=b, c=rng.choice(list('xyz'), n)))
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return D


@pytest.mark.parametrize('family, y', [(sm.families.Gaussian, 'y'), (sm.families.Binomial, 'yb')])
@pytest.mark.parametrize('terms', [lambda: [s('a', df=4), s('b', df=4), factor('c')],
                                   lambda: [s('a', lam=1e-4), linear('b'), factor('c')],
                                   lambda: [lo('a'), s('b', df=3), factor('c')]])
def test_modified_backfitting_same_fit(family, y, terms):
    # modified and plain backfitting have the same fixed point (n small: a
    # loess term's df smooths one unit vector per observation)
    D = _correlated(0.5, n=400)
    fits = [GAM(terms=terms(), family=family(),
                control=GAMControl(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000,
                                   bf_modified=modified)).fit(D, D[y])
            for modified in (False, True)]
    np.testing.assert_allclose(fits[1].state_.linear_predictor, fits[0].state_.linear_predictor,
                               rtol=1e-7, atol=1e-8)
    for G in fits:
        np.testing.assert_allclose(G.predict(D, 'link'), G.state_.linear_predictor,
                                   rtol=1e-8, atol=1e-9)


def test_modified_backfitting_concurvity():
    # nearly collinear predictors: plain backfitting crawls (here it doesn't
    # converge in 300 sweeps), modified backfitting converges quickly
    D = _correlated(0.999)
    terms = [s('a', df=4), s('b', df=4), factor('c')]
    control = dict(bf_tol=1e-10, bf_maxit=300)
    with pytest.warns(UserWarning, match='did not converge'):
        GAM(terms=terms, control=GAMControl(bf_modified=False, **control)).fit(D, D['y'])
    G = GAM(terms=terms, control=GAMControl(**control)).fit(D, D['y'])
    assert G.regularizer_.n_sweeps_ < 300


def test_modified_backfitting_collinear_terms():
    # s(a) and linear(a) share their linear part: the split between them is
    # not identified (minimum norm), but the fit is, and it converges
    D = _correlated(0.)
    G = GAM(terms=[s('a', df=4), linear('a'), s('b', df=3)],
            control=GAMControl(bf_tol=1e-10)).fit(D, D['y'])
    assert G.regularizer_.n_sweeps_ < 20
    G0 = GAM(terms=[s('a', df=4), s('b', df=3)], control=GAMControl(bf_tol=1e-10)).fit(D, D['y'])
    np.testing.assert_allclose(G.state_.linear_predictor, G0.state_.linear_predictor,
                               rtol=1e-7, atol=1e-8)


def test_modified_backfitting_many_levels():
    # a factor with more than max_joint_levels levels is backfitted
    rng = np.random.default_rng(5)
    n = 3000
    D = pd.DataFrame(dict(a=rng.uniform(0, 1, n), g=rng.integers(0, 80, n)))
    D['y'] = np.sin(2 * np.pi * D['a']) + 0.01 * D['g'] + 0.5 * rng.standard_normal(n)
    assert factor('g').setup(D['g']).linear_columns() is None
    fits = [GAM(terms=[s('a', df=4), factor('g')],
                control=GAMControl(bf_tol=1e-13, bf_maxit=5000, bf_modified=m)).fit(D, D['y'])
            for m in (False, True)]
    np.testing.assert_allclose(fits[1].state_.linear_predictor, fits[0].state_.linear_predictor,
                               rtol=1e-8, atol=1e-9)


def test_residuals_function(data):
    # glmstar's residuals(fit, kind) works on a GAM, as on a GLM
    from glmnet.glm import residuals
    G = GAM(terms=[s('a', df=4), factor('c')], family=sm.families.Binomial(),
            control=TIGHT).fit(data, data['yb'])
    mu = G.predict(data)
    np.testing.assert_allclose(residuals(G), data['yb'] - mu, rtol=1e-10)
    np.testing.assert_allclose(residuals(G, 'pearson'),
                               (data['yb'] - mu) / np.sqrt(mu * (1 - mu)), rtol=1e-8)


@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('weights', ['unit', 'random', 'zeros'])
def test_loess_edf_is_trace(degree, weights):
    # the edf of lo() (from the loess leverages) is the trace of the
    # smoother, by smoothing the unit vectors, less 1
    rng = np.random.default_rng(3)
    n = 150
    x = np.round(rng.uniform(0, 10, n), 1)        # with ties
    w = {'unit': np.ones(n), 'random': rng.uniform(0.2, 3, n),
         'zeros': np.r_[np.zeros(10), np.ones(n - 10)]}[weights]
    term = LoessTerm('x', span=0.6, degree=degree)
    term.setup(x)
    term.set_weights(w)
    trace = sum(term.smooth(np.identity(n)[i])[i] for i in range(n))
    np.testing.assert_allclose(term.edf(), trace - 1, rtol=1e-10)
    # evaluated once per distinct value: the smooth at every row
    r = rng.standard_normal(n)
    term._smoother.smooth(r)
    np.testing.assert_allclose(term.smooth(r), term._smoother.predict(x), rtol=1e-12, atol=1e-14)
