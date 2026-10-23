"""
Varying-coefficient terms (`varying(term, by=...)`, mgcv's `by=`) and the
cubic regression spline (`cubic_regression_spline()`, mgcv's `bs="cr"`):
the per-level constraints, backfitting against the joint solve, identities
(a pooled random effect by a categorical and the interaction random
effect, `lo()` by a categorical and separate loess fits), the REML
gradient, prediction, plots, the display, and against mgcv (factor and
numeric `by`, with `s()`, cyclic splines and tensor products; pooled
smoothing parameters, mgcv's `id`).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

pytest.importorskip('scatter_smooth')

from glmnet.gam import (GAM, GLMM, GAMControl, s, lo, random_effect as re, categorical, linear, varying, cyclic_spline,
                        tensor_spline, cubic_regression_spline)

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(5)
    n = 480
    # (x on 30 distinct values: s() has a knot at each, as mgcv given them)
    x = np.round(rng.uniform(0, 1, n) * 29) / 29
    f = rng.choice(['a', 'b', 'c'], n)
    D = pd.DataFrame({'x': x, 'f': f, 'z': rng.uniform(0.5, 2, n),
                      'u': rng.uniform(0, 1, n), 'h': rng.uniform(0, 24, n),
                      'g': rng.choice([f'g{i}' for i in range(8)], n)})
    curve = np.where(f == 'a', np.sin(2 * np.pi * x), np.where(f == 'b', 2 * x**2, 0.5 * np.cos(4 * x)))
    daily = np.where(f == 'a', np.sin(2 * np.pi * D['h'] / 24), 0.5 * np.cos(2 * np.pi * D['h'] / 24))
    D['y'] = (curve + (f == 'b') + D['z'] * np.sin(3 * x) + daily + np.sin(3 * D['u'])
              + 0.3 * rng.standard_normal(n))
    D['yp'] = rng.poisson(np.exp(0.3 + 0.5 * curve + 0.3 * daily))
    return D


def _fit(D, terms, family='gaussian', method='fixed', solver='backfit'):
    fam, y = (sm.families.Gaussian(), 'y') if family == 'gaussian' else (sm.families.Poisson(), 'yp')
    return GLMM(terms=terms, family=fam, method=method,
               control=GAMControl(solver=solver, **TIGHT)).fit(D, D[y])


# the terms, and the constraints

def test_level_curves(data):
    G = _fit(data, [categorical('f'), varying(s('x'), by='f', lam=1e-3)])
    term = G.terms_[1]
    assert term.is_categorical_ and list(term.levels_) == ['a', 'b', 'c']
    fit = term.predict(data[['x', 'f']].to_numpy())
    for level in term.levels_:
        rows = (data['f'] == level).to_numpy()
        # zero off its level; summing to 0 over all the rows' x, as mgcv's
        as_level = np.column_stack([data['x'], np.full(len(data), level, dtype=object)])
        np.testing.assert_allclose(term.predict(as_level).sum(), 0, atol=1e-9)
        other = np.column_stack([data['x'], np.where(rows, level, 'zz')])
        assert np.all(term.predict(other)[~rows] == 0)
        np.testing.assert_allclose(term.predict(other)[rows], fit[rows], rtol=1e-12)
    np.testing.assert_allclose(term.level_edf().sum(), G.edf_.iloc[1], rtol=1e-10)
    assert term.name == 'varying(s(x), by=f, lam=0.001)'


def test_numeric_by(data):
    G = _fit(data, [varying(s('x'), by='z', lam=1e-3, as_categorical=False)])
    term = G.terms_[0]
    assert not term.is_categorical_
    # z f(x): the coefficient function at z = 1, times z
    f1 = term.predict(np.column_stack([data['x'], np.ones(len(data))]))
    np.testing.assert_allclose(term.predict(data[['x', 'z']].to_numpy()), data['z'] * f1, rtol=1e-12)
    # not centered: the coefficient function has its own level
    assert abs(f1.mean()) > 0.1


def test_by_is_categorical_by_default(data):
    # a categorical coded 0 and 1 (or any numbers) is categorical unless as_categorical=False
    D = data.assign(k=(data['f'] == 'a').astype(int))
    G = _fit(D, [categorical('k'), varying(s('x'), by='k', lam=1e-3)])
    assert G.terms_[1].is_categorical_ and list(G.terms_[1].levels_) == [0, 1]
    H = _fit(D, [varying(s('x'), by='k', lam=1e-3, as_categorical=False)])
    assert not H.terms_[0].is_categorical_
    assert H.terms_[0].name == 'varying(s(x), by=k, lam=0.001, as_categorical=False)'


def test_inner_lam(data):
    # the inner term's lam is every level's
    G = _fit(data, [categorical('f'), varying(s('x', lam=1e-3), by='f')])
    H = _fit(data, [categorical('f'), varying(s('x'), by='f', lam=1e-3)])
    np.testing.assert_allclose(G.predict(data), H.predict(data), rtol=1e-12)
    assert G.terms_[1].lam == [1e-3] * 3
    # a tensor product's lams (one per margin) for each level, or pooled
    T = _fit(data, [categorical('f'), varying(tensor_spline('x', 'u', k=(5, 4), lam=[1e-3, 1e-2]),
                                         by='f')], solver='joint')
    assert T.terms_[1].lam == [1e-3, 1e-2] * 3
    P = _fit(data, [categorical('f'), varying(tensor_spline('x', 'u', k=(5, 4), lam=[1e-3, 1e-2]),
                                         by='f', pooled_lam=True)], solver='joint')
    np.testing.assert_allclose(P.predict(data), T.predict(data), rtol=1e-10)


@pytest.mark.parametrize('inner', ['s', 'cr', 'cyclic', 'tensor'])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_backfit_is_joint(data, inner, family):
    make = {'s': lambda: s('x', lam=1e-3),
            'cr': lambda: cubic_regression_spline('x', lam=1e-3),
            'cyclic': lambda: cyclic_spline('h', period=(0, 24), lam=1e-3),
            'tensor': lambda: tensor_spline('x', 'u', k=(5, 4), lam=[1e-3, 1e-2])}[inner]
    # (another smooth, of a column the varying term does not use: a tensor
    # product's curves per level, summed, have s('u')'s linear part, a
    # direction the fit does not identify, which the two solvers' fits
    # resolve differently by platform)
    other = s('h', lam=1e-2) if inner == 'tensor' else s('u', lam=1e-2)
    for by in ['f', 'z']:
        terms = [categorical('f'), varying(make(), by=by, as_categorical=by == 'f'), other]
        fits = [_fit(data, terms, family, solver=solver) for solver in ['backfit', 'joint']]
        np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-7)
        np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)


def test_pooled_random_effect_is_interaction(data):
    # re(g) by a categorical, the levels sharing the variance: the random effect
    # of the combinations of g and f
    G = _fit(data, [categorical('f'), varying(re('g'), by='f', lam=0.05, pooled_lam=True)],
             solver='joint')
    H = _fit(data, [categorical('f'), re(['g', 'f'], lam=0.05)], solver='joint')
    np.testing.assert_allclose(G.predict(data), H.predict(data), rtol=1e-10)
    np.testing.assert_allclose(G.edf_.sum(), H.edf_.sum(), rtol=1e-10)


def test_loess_by_categorical(data):
    # lo() by a categorical: a loess per level, fitted to that level's data
    # (the categorical gives the means)
    G = GAM(terms=[categorical('f'), varying(lo('x', span=0.6), by='f')],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.converged_
    for level in ['a', 'b', 'c']:
        part = data[data['f'] == level]
        H = GAM(terms=[lo('x', span=0.6)], control=GAMControl(**TIGHT)).fit(part, part['y'])
        np.testing.assert_allclose(G.predict(part), H.predict(part), rtol=1e-8)
    # its edf, a loess's per level
    np.testing.assert_allclose(G.terms_[1].level_edf().sum(), G.edf_.iloc[1], rtol=1e-10)


def test_loess_by_number(data):
    # Hastie and Tibshirani's update: z times the loess of r / z with weights w z^2
    G = GAM(terms=[varying(lo('x', span=0.6), by='z', as_categorical=False), s('u', df=3)],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.converged_
    term = G.terms_[0]
    np.testing.assert_allclose(term.predict(data[['x', 'z']].to_numpy()), G.fits_.iloc[:, 0],
                               atol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('pooled', [False, True])
def test_gradient(data, method, pooled):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    G = _fit(data, [categorical('f'), varying(s('x'), by='f', pooled_lam=pooled), s('u')],
             'poisson', method)
    assert G.smoothing_.converged
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [1, 2], method=method)
        assert C.n_params == (1 if pooled else 3) + 1
        x = C.start() + np.random.default_rng(0).uniform(-0.5, 0.5, C.n_params)
        _, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = [(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
              for e in np.identity(C.n_params)]
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


def test_predict_and_order(data):
    D = data.assign(f=pd.Categorical(data['f'], categories=['c', 'a', 'b', 'd']))
    G = _fit(D, [categorical('f'), varying(s('x'), by='f', lam=[1e-4, 1e-3, 1e-2])])
    H = _fit(data, [categorical('f'), varying(s('x'), by='f', lam=[1e-3, 1e-2, 1e-4])])
    assert list(G.terms_[1].levels_) == ['c', 'a', 'b']
    np.testing.assert_allclose(G.predict(D), H.predict(data), rtol=1e-8)
    # new data: the fit, and 0 for a level not seen
    np.testing.assert_allclose(G.predict(D.iloc[:7]), G.predict(D)[:7], rtol=1e-12)
    new = D.iloc[:5].assign(f=pd.Categorical(['d'] * 5, categories=['c', 'a', 'b', 'd']))
    np.testing.assert_allclose(G.partial_effect(1, new), 0)


def test_errors(data):
    with pytest.raises(ValueError, match='not df'):
        varying(s('x', df=4), by='f')
    with pytest.raises(ValueError, match='one per penalty of each level'):
        _fit(data, [varying(s('x'), by='f', lam=[1., 2.])])
    with pytest.raises(ValueError, match='give lam'):
        _fit(data, [varying(s('x'), by='f')])
    with pytest.raises(ValueError, match='need a basis'):
        _fit(data, [categorical('f'), varying(lo('x'), by='f'), s('u')], method='REML')
    with pytest.raises(ValueError, match='has no basis'):
        _fit(data, [varying(lo('x'), by='f', lam=1.)])


def test_plot(data):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    G = _fit(data, [categorical('f'), varying(s('x'), by='f', lam=1e-3), varying(s('u'), by='z', lam=1e-2, as_categorical=False),
                    varying(tensor_spline('x', 'u', k=(4, 4), lam=[1., 1.]), by='f')], solver='joint')
    axes = G.plot()
    assert len(axes) == 3                         # (not the tensor product's)
    assert len(axes[1].get_lines()) == 4          # a curve per level, and the rug
    with pytest.raises(ValueError, match='several columns'):
        G.plot(terms=[3])


def test_display(data):
    g = GLMM([categorical('f'), varying(s('x'), by='f', pooled_lam=True),
             varying(s('u'), by='z', as_categorical=False), varying(lo('h'), by='f'),
             cubic_regression_spline('u')], method='REML')
    latex = g.model_latex()
    assert r'f_{2,\text{f}[i]}(\text{x}_i)' in latex
    assert 'one smoothing parameter for all the levels' in latex
    assert r'\text{z}_i\, f_{3}(\text{u}_i)' in latex
    assert 'cubic regression spline, 10 knots' in latex
    # (loess has no smoothing parameters to share)
    assert latex.count('smoothing parameter') == 1


# against mgcv

def _R(Rinfo, D):
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    rpy.r('D$f <- factor(D$f)')
    return rpy


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))


def _compare(Rinfo, G, family, method, n_smooth_params, exact):
    """Our criterion at most its value at mgcv's smoothing parameters; the fits close."""
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    fitted = _get(Rinfo, 'as.numeric(predict(m))')
    rho_R = np.log(_get(Rinfo, 'm$sp') /
                   _get(Rinfo, 'unlist(lapply(m$smooth, function(s) rep(s$S.scale, length(s$S))))'
                        )[:len(_get(Rinfo, 'm$sp'))])
    rho = np.log(G.smoothing_.lam_raw)
    estimated = [j for j, t in enumerate(G.terms_) if t.name in G.smoothing_.terms or
                 any(n.startswith(t.name + '[') for n in G.smoothing_.terms)]
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, estimated, method=method)

        def value(r):
            if C.n_params == C.n_rho:
                return float(C.value_and_grad(r)[0])
            return minimize_scalar(lambda lp: float(C.value_and_grad(np.r_[r, lp])[0]),
                                   bounds=(-10, 5), method='bounded').fun

        assert rho_R.shape == rho.shape == (n_smooth_params,)
        # (mgcv goes further along a flat ridge, a smoothing parameter
        # going to infinity: its criterion can be lower, by about 1e-8)
        v, v_R = value(rho), value(rho_R)
        assert v <= v_R + 1e-7 * abs(v_R)
    finite = rho_R < 10
    np.testing.assert_allclose(rho[finite], rho_R[finite], atol=0.1)
    if method == 'ML':
        # (no constant of the parametrization: the same criterion as mgcv's;
        # mgcv can go further along a flat ridge, about 3e-8)
        score = _get(Rinfo, 'm$gcv.ubre')[0]
        assert G.reml_score_ <= score + 1e-7 * abs(score)
        np.testing.assert_allclose(G.reml_score_, score, rtol=1e-7)
    np.testing.assert_allclose(G.state_.link_parameter, fitted, atol=1e-7 if exact else 1e-2)


@pytest.mark.parametrize('pooled', [False, True])
@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_mgcv_categorical_by(Rinfo, data, family, method, pooled):
    rpy = _R(Rinfo, data)
    G = _fit(data, [categorical('f'), varying(s('x'), by='f', pooled_lam=pooled)], family, method)
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kn', G.terms_[1].inner_.knots_)
    y, Rf = ('y', 'gaussian()') if family == 'gaussian' else ('yp', 'poisson()')
    idp = ', id=1' if pooled else ''
    rpy.r(f"""m <- mgcv::gam({y} ~ f + s(x, by=f, bs='cr', k=length(kn){idp}), data=D,
                           family={Rf}, method='{method}', knots=list(x=kn),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    _compare(Rinfo, G, family, method, 1 if pooled else 3,
             exact=family == 'gaussian' and method == 'REML')


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_mgcv_numeric_by(Rinfo, data, family, method):
    rpy = _R(Rinfo, data)
    G = _fit(data, [varying(s('x'), by='z', as_categorical=False), s('u')], family, method)
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kn', G.terms_[0].inner_.knots_)
        rpy.r.assign('ku', G.terms_[1].knots_)
    y, Rf = ('y', 'gaussian()') if family == 'gaussian' else ('yp', 'poisson()')
    rpy.r(f"""m <- mgcv::gam({y} ~ s(x, by=z, bs='cr', k=length(kn)) + s(u, bs='cr', k=length(ku)),
                           data=D, family={Rf}, method='{method}', knots=list(x=kn, u=ku),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    _compare(Rinfo, G, family, method, 2, exact=family == 'gaussian' and method == 'REML')


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_mgcv_cyclic_by(Rinfo, data, method):
    rpy = _R(Rinfo, data)
    G = _fit(data, [categorical('f'), varying(cyclic_spline('h', k=8, period=(0, 24)), by='f')],
             'gaussian', method)
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kh', G.terms_[1].inner_.spline_.knots)
    rpy.r(f"""m <- mgcv::gam(y ~ f + s(h, by=f, bs='cc', k=8), data=D, method='{method}',
                           knots=list(h=kh),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    _compare(Rinfo, G, 'gaussian', method, 3, exact=method == 'REML')


def test_mgcv_tensor_by(Rinfo, data):
    # (mgcv's te sp are on the scale of its reparametrization: the ML
    # criteria, which have no constant of the parametrization, and the fits)
    rpy = _R(Rinfo, data)
    G = _fit(data, [categorical('f'), varying(tensor_spline('x', 'u', k=(5, 4)), by='f')],
             'gaussian', 'ML')
    assert G.smoothing_.converged
    margins = G.terms_[1].inner_.margins_
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kx', margins[0].knots)
        rpy.r.assign('ku', margins[1].knots)
    rpy.r("""m <- mgcv::gam(y ~ f + te(x, u, by=f, bs='cr', k=c(5, 4)), data=D, method='ML',
                           knots=list(x=kx, u=ku),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    score = _get(Rinfo, 'm$gcv.ubre')[0]
    assert G.reml_score_ <= score + 1e-8 * abs(score)
    np.testing.assert_allclose(G.reml_score_, score, rtol=1e-9)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'as.numeric(predict(m))'),
                               atol=1e-3)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_mgcv_cubic_regression_spline(Rinfo, data, method):
    rpy = _R(Rinfo, data)
    G = _fit(data, [cubic_regression_spline('u', k=12), categorical('f')], 'poisson', method)
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('ku', G.terms_[0].spline_.knots)
    rpy.r(f"""m <- mgcv::gam(yp ~ s(u, bs='cr', k=12) + f, data=D, family=poisson(),
                           method='{method}', knots=list(u=ku),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    _compare(Rinfo, G, 'poisson', method, 1, exact=False)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'as.numeric(predict(m))'),
                               atol=1e-5)
