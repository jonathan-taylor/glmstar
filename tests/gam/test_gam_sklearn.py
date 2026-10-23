"""
GAM and GLMM as scikit-learn estimators: clone, the terms' parameters
(`terms__<j>`, `terms__<j>__<param>`; a GLMM leaves out those it
estimates), `score` (minus half the held-out deviance; the partial
likelihood for Cox), and tuning by GridSearchCV: the smoother of a column,
its degrees of freedom, a basis' size.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, cross_val_score
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor

pytest.importorskip('scatter_smooth')

from glmnet.gam import (GAM, GLMM, GAMControl, ExperimentalWarning, s, random_effect as re, linear, categorical,
                        tensor_spline as te, gaussian_process, varying, TransformerSmoother,
                        RegressorSmoother)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 400
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.standard_normal(n),
                      'g': rng.integers(0, 20, n), 'c': rng.choice(list('ab'), n),
                      'w': rng.uniform(0.5, 2, n)})
    b = rng.normal(0, 0.5, 20)
    D['y'] = np.sin(6 * D['x']) + 0.3 * D['z'] + b[D['g']] + 0.3 * rng.standard_normal(n)
    D['count'] = rng.poisson(np.exp(0.5 * np.sin(6 * D['x'])))
    return D


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        yield


def test_clone(data):
    G = GAM([s('x', df=5), linear('z'), re('g', lam=0.1)])
    H = clone(G)
    assert H.terms[0] is not G.terms[0]
    assert [t.name for t in H.terms] == [t.name for t in G.terms]
    np.testing.assert_allclose(H.fit(data, data['y']).predict(data),
                               G.fit(data, data['y']).predict(data))


def test_params(data):
    G = GAM([s('x', df=5), linear('z')])
    params = G.get_params()
    assert params['terms__0__df'] == 5 and params['terms__0'] is G.terms[0]
    first = G.terms[0]
    G.set_params(terms__0__df=7)
    assert G.terms[0].df == 7 and G.terms[0].name == 's(x, df=7)'
    assert first.df == 5        # (the term given is not changed)
    G.set_params(terms__0=s('x', lam=1e-3))
    assert G.terms[0].name == 's(x, lam=0.001)'
    with pytest.raises(ValueError, match='one of df and lam'):
        GAM([s('x', df=5)]).set_params(terms__0__lam=1.)
    with pytest.raises(ValueError):
        G.set_params(terms__0__nope=1)
    # nested: a varying term's inner term
    V = GAM([varying(s('x', lam=1.), by='c')])
    assert V.get_params()['terms__0__inner__lam'] == 1.
    V.set_params(terms__0__inner__lam=2.)
    assert V.terms[0].inner.lam == 2.
    # what a smoother derives from its arguments follows them
    P = GAM([gaussian_process('x', k=10, lam=1., range=0.2)])
    P.set_params(terms__0__range=0.5)
    assert P.terms[0].range_given_ == 0.5 or P.terms[0].range == 0.5


def test_glmm_params(data):
    G = GLMM([s('x'), s('z', lam=1.), re('g'), te('x', 'z', k=5)])
    params = G.get_params()
    # estimated: not parameters to tune; given: tunable
    assert 'terms__0__lam' not in params and 'terms__0__df' not in params
    assert params['terms__1__lam'] == 1.
    assert 'terms__2__lam' not in params
    assert params['terms__3__k'] == 5
    assert 'terms__0__column' in params


def test_score(data):
    G = GAM([s('x', df=5), linear('z'), categorical('c')]).fit(data, data['y'])
    mu = G.predict(data)
    np.testing.assert_allclose(G.score(data, data['y']), -np.sum((data['y'] - mu)**2) / 2)
    W = GAM([s('x', df=5), linear('z')], response_id='y', weight_id='w').fit(data, data[['y', 'w']])
    mu = W.predict(data)
    np.testing.assert_allclose(W.score(data, data[['y', 'w']]),
                               -np.sum(data['w'] * (data['y'] - mu)**2) / 2)
    P = GAM([s('x', df=5)], family=sm.families.Poisson()).fit(data, data['count'])
    mu = P.predict(data)
    y = data['count'].to_numpy()
    dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
    np.testing.assert_allclose(P.score(data, data['count']), -dev / 2, rtol=1e-10)


def test_grid_search_smoothers(data):
    # which smoother of x: a spline, a P-spline, a tree
    G = GAM([s('x', df=5), linear('z')])
    grid = {'terms__0': [s('x', df=6),
                         TransformerSmoother('x', SplineTransformer(n_knots=15), lam=1e-3),
                         RegressorSmoother('x', DecisionTreeRegressor(max_depth=2), df=4)]}
    search = GridSearchCV(G, grid, cv=KFold(5, shuffle=True, random_state=0)).fit(data, data['y'])
    scores = search.cv_results_['mean_test_score']
    assert np.all(np.isfinite(scores))
    # a depth-2 tree is a poor fit of a sine
    assert np.argmin(scores) == 2
    assert search.best_estimator_.terms_[0].name in ('s(x, df=6)', 'transformer(x)')


def test_grid_search_df(data):
    search = GridSearchCV(GAM([s('x', df=4), linear('z')]), {'terms__0__df': [2, 4, 8]},
                          cv=KFold(5, shuffle=True, random_state=0)).fit(data, data['y'])
    assert search.best_params_['terms__0__df'] in (4, 8)
    assert search.best_estimator_.terms_[0].name == f"s(x, df={search.best_params_['terms__0__df']})"


def test_glmm_cv(data):
    pytest.importorskip('jax')
    # a basis' size, the smoothing parameter estimated by REML in each fold;
    # GroupKFold: the score of prediction for new groups
    G = GLMM([te('x', 'z', k=4), re('g')], control=GAMControl(mxitnr=50))
    search = GridSearchCV(G, {'terms__0__k': [4, 6]},
                          cv=GroupKFold(4)).fit(data, data['y'], groups=data['g'])
    assert np.all(np.isfinite(search.cv_results_['mean_test_score']))
    assert search.best_estimator_.smoothing_.converged
    scores = cross_val_score(GLMM([s('x'), linear('z'), re('g')]), data, data['y'],
                             cv=KFold(4, shuffle=True, random_state=1))
    assert np.all(np.isfinite(scores))


def test_cox_score(data):
    from glmnet.cox import CoxFamily
    rng = np.random.default_rng(2)
    D = data.assign(time=rng.exponential(np.exp(-np.sin(6 * data['x']))),
                    status=rng.integers(0, 2, len(data)) | 1)
    fam = CoxFamily(event_id='time', status_id='status')
    scores = cross_val_score(GAM([s('x', df=4)], family=fam), D, D[['time', 'status']],
                             cv=KFold(4, shuffle=True, random_state=0))
    null = cross_val_score(GAM([s('z', df=1.5)], family=fam), D, D[['time', 'status']],
                           cv=KFold(4, shuffle=True, random_state=0))
    assert np.all(np.isfinite(scores))
    assert scores.mean() > null.mean()
