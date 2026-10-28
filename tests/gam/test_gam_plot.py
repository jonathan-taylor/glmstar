"""
`GAM.plot`, `plot_term`: the drawn curves and bands are the terms' fits and
their pointwise standard errors.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from glmnet.cox import CoxFamily
from glmnet.gam import GAM, GAMControl, ColumnsSmoother, s, lo, random_effect as re, linear, categorical, plot_term


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 300
    D = pd.DataFrame({'a': rng.uniform(0, 1, n),
                      'b': rng.normal(0, 1, n),
                      'c': rng.choice(list('xyz'), n),
                      'g': rng.choice([f'g{i}' for i in range(8)], n),
                      'v': rng.uniform(0, 2, n)})
    D['y'] = (np.sin(2 * np.pi * D['a']) + 0.5 * D['b'] + D['c'].map({'x': 0, 'y': .5, 'z': -.5}) +
              rng.standard_normal(n))
    D['b2'] = D['b']**2
    return D


@pytest.fixture(autouse=True)
def close():
    yield
    plt.close('all')


def _lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() in ('-', '--')]


def test_curve_and_bands(data):
    G = GAM(terms=[s('a', df=5), linear('b'), categorical('c')]).fit(data, data['y'])
    ax = plot_term(G, 's(a, df=5)')
    curve, lower, upper = [l for l in ax.get_lines() if l.get_marker() in ('None', '')][:3]
    grid = curve.get_xdata()
    np.testing.assert_allclose(grid[[0, -1]], [data['a'].min(), data['a'].max()])
    fit, std = G.partial_effect('s(a, df=5)', grid, return_std=True)
    np.testing.assert_allclose(curve.get_ydata(), fit)
    np.testing.assert_allclose(lower.get_ydata(), fit - 2 * std)
    np.testing.assert_allclose(upper.get_ydata(), fit + 2 * std)
    assert lower.get_linestyle() == '--'
    # the rug: the distinct training values
    rug = [l for l in ax.get_lines() if l.get_marker() == '|'][0]
    np.testing.assert_allclose(rug.get_xdata(), np.unique(data['a']))
    assert ax.get_ylabel() == 's(a, df=5)' and ax.get_xlabel() == 'a'
    # a linear term: its line, with bands
    ax = plot_term(G, 1, se=False, rug=False)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), G.partial_effect(1, line.get_xdata()))


def test_factor_and_random_effect(data):
    G = GAM(terms=[s('a', df=4), categorical('c'), re('g', lam=0.1), re('g', by='v', lam=1.)],
            control=GAMControl(solver='joint')).fit(data, data['y'])
    ax = plot_term(G, 'categorical(c)')
    points = ax.get_lines()[0]
    fit, std = G.partial_effect('categorical(c)', np.array(['x', 'y', 'z']), return_std=True)
    np.testing.assert_allclose(points.get_ydata(), fit)
    assert [t.get_text() for t in ax.get_xticklabels()] == ['x', 'y', 'z']
    assert len(ax.collections) == 1                  # the error bars
    # random intercepts, and slopes (the effect of v = 1)
    for name in ['random_effect(g, lam=0.1)', 'random_effect(g, by=v, lam=1.0)']:
        ax = plot_term(G, name)
        j = [t.name for t in G.terms_].index(name)
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), G.terms_[j].effects_.values)
        assert len(ax.get_xticklabels()) == 8


def test_loess_has_no_bands(data):
    G = GAM(terms=[lo('a', span=0.4), linear('b')]).fit(data, data['y'])
    assert G.posterior_covariance_ is None
    ax = plot_term(G, 0)
    assert len([l for l in ax.get_lines() if l.get_linestyle() == '--']) == 0
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(),
                               G.partial_effect(0, ax.get_lines()[0].get_xdata()))


def test_plot_method(data):
    G = GAM(terms=[s('a', df=4), linear('b'), categorical('c'), ColumnsSmoother(['b', 'b2'])]).fit(
        data, data['y'])
    axes = G.plot()
    # the block of two columns has no curve
    assert len(axes) == 3
    assert [ax.get_ylabel() for ax in axes] == ['s(a, df=4)', 'linear(b)', 'categorical(c)']
    axes = G.plot(terms=['categorical(c)'], se=False)
    assert len(axes) == 1
    fig, given = plt.subplots(1, 2)
    out = G.plot(terms=[0, 1], axes=given)
    assert list(out) == list(given)
    with pytest.raises(ValueError, match='block of several columns'):
        plot_term(G, 3)
    with pytest.raises(ValueError, match='not a term'):
        plot_term(G, 's(b)')
    with pytest.raises(ValueError, match='axes given'):
        G.plot(axes=given)


def test_plot_cox(data):
    rng = np.random.default_rng(1)
    D = data.copy()
    T = rng.exponential(np.exp(-np.sin(2 * np.pi * D['a'])))
    D['time'], D['status'] = np.minimum(T, 2.), (T <= 2.).astype(int)
    G = GAM(terms=[s('a', df=4), linear('b')], family=CoxFamily(event_id='time', status_id='status'),
            control=GAMControl(solver='joint')).fit(D, D)
    ax = plot_term(G, 0)
    curve, lower, upper = ax.get_lines()[:3]
    fit, std = G.partial_effect(0, curve.get_xdata(), return_std=True)
    np.testing.assert_allclose(upper.get_ydata() - lower.get_ydata(), 4 * std)


# fixed effects of one column of the data: drawn from their columns rebuilt
# at a grid of that column, or at its levels

def test_fixed_effect_curve(data):
    from modelmatrix import ns, poly
    D = data.assign(c=data['c'].astype('category'))
    G = GAM(terms=[ns('a', df=4), poly('b', 2), 'c']).fit(D, D['y'])
    ax = plot_term(G, 'ns(a, df=4)')
    curve, lower, upper = [l for l in ax.get_lines() if l.get_marker() in ('None', '')][:3]
    grid = curve.get_xdata()
    np.testing.assert_allclose(grid[[0, -1]], [data['a'].min(), data['a'].max()])
    # the term's columns at the grid, times its coefficients, centered as the term
    term = G.terms_[0]
    B = np.asarray(G.model_specs_[0].build_submodel(pd.DataFrame({'a': grid}), [term.term_]))
    np.testing.assert_allclose(curve.get_ydata(), (B - term.center_) @ term.coef_, atol=1e-10)
    fit, std = G.partial_effect(0, B, return_std=True)
    np.testing.assert_allclose(lower.get_ydata(), fit - 2 * std)
    np.testing.assert_allclose(upper.get_ydata(), fit + 2 * std)
    rug = [l for l in ax.get_lines() if l.get_marker() == '|'][0]
    np.testing.assert_allclose(rug.get_xdata(), np.unique(data['a']))
    assert ax.get_xlabel() == 'a' and ax.get_ylabel() == 'ns(a, df=4)'
    assert plot_term(G, 1).get_xlabel() == 'b'
    # every term has a panel
    assert [ax.get_xlabel() for ax in G.plot()] == ['a', 'b', 'c']


def test_fixed_effect_levels(data):
    from modelmatrix import contrast
    D = data.assign(c=data['c'].astype('category'))
    G = GAM(terms=[s('a', df=4), 'c']).fit(D, D['y'])
    H = GAM(terms=[s('a', df=4), categorical('c')]).fit(D, D['y'])
    ax, bx = plot_term(G, 'c'), plot_term(H, 'categorical(c)')
    # treatment coding, centered: the effects of categorical()
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), bx.get_lines()[0].get_ydata(),
                               atol=1e-8)
    assert [t.get_text() for t in ax.get_xticklabels()] == ['x', 'y', 'z']
    assert ax.get_xlabel() == 'c' and len(ax.collections) == 1
    # a column of strings, and a contrast
    G = GAM(terms=[s('a', df=4), contrast('c', 'sum')]).fit(data, data['y'])
    ax = plot_term(G, 1)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), bx.get_lines()[0].get_ydata(),
                               atol=1e-8)


def test_fixed_effect_of_a_feature(data):
    # a function of a feature of one column (ns(log(a))): drawn against the column
    from modelmatrix import ns, derived_feature
    from sklearn.preprocessing import FunctionTransformer
    log_a = derived_feature(['a'], encoder=FunctionTransformer(np.log), name='log(a)')
    G = GAM(terms=[ns(log_a, df=3), 'b']).fit(data, data['y'])
    ax = plot_term(G, 0)
    assert ax.get_xlabel() == 'a'
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata()[[0, -1]],
                               [data['a'].min(), data['a'].max()])


def test_fixed_effect_of_several_columns(data):
    from modelmatrix import pca
    D = data.assign(c=data['c'].astype('category'))
    G = GAM(terms=[s('a', df=4), ('b', 'c'), pca(['b', 'v'], n_components=2, name='pc')]).fit(
        D, D['y'])
    assert [ax.get_ylabel() for ax in G.plot()] == ['s(a, df=4)']
    with pytest.raises(ValueError, match=r"function of the columns \['b', 'c'\]"):
        plot_term(G, 'b:c')
    with pytest.raises(ValueError, match=r"function of the columns \['b', 'v'\]"):
        plot_term(G, 2)


def test_fixed_effect_pickled(data):
    import pickle
    from modelmatrix import ns
    D = data.assign(c=data['c'].astype('category'))
    G = GAM(terms=[ns('a', df=4), 'c']).fit(D, D['y'])
    P = pickle.loads(pickle.dumps(G))
    for j in range(2):
        np.testing.assert_allclose(plot_term(P, j).get_lines()[0].get_ydata(),
                                   plot_term(G, j).get_lines()[0].get_ydata())
