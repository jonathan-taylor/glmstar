"""
Cyclic smooths (`cyclic_spline()`, mgcv's `s(x, bs="cc")`): the cyclic cubic
regression spline basis (its continuity around the cycle and its penalty),
backfitting against the joint solve, prediction around the cycle, and
against mgcv under REML and ML.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GLMM, GAMControl, s, cyclic_spline
from glmnet.gam.cyclic import CyclicCubicSpline

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 500
    D = pd.DataFrame({'hour': rng.uniform(0, 24, n), 'x': rng.uniform(0, 1, n)})
    D['y'] = (np.sin(2 * np.pi * D['hour'] / 24) + 0.5 * np.cos(4 * np.pi * D['hour'] / 24) +
              np.sin(3 * D['x']) + 0.3 * rng.standard_normal(n))
    D['yp'] = rng.poisson(np.exp(0.5 + 0.5 * np.sin(2 * np.pi * D['hour'] / 24)))
    return D


def test_basis():
    rng = np.random.default_rng(1)
    t = np.sort(np.r_[0, rng.uniform(0, 24, 7), 24])
    spline = CyclicCubicSpline(t)
    m = len(t) - 1
    # the coefficients are the values at the first k - 1 knots; the basis sums to 1
    np.testing.assert_allclose(spline.basis(t[:-1]), np.identity(m), atol=1e-14)
    np.testing.assert_allclose(spline.basis(np.linspace(0, 24, 101)).sum(1), 1, atol=1e-14)
    # periodic: the function, and its first two derivatives, meet at the ends
    c = rng.standard_normal(m)
    f = lambda x: spline.basis(np.atleast_1d(x)) @ c
    np.testing.assert_allclose(f(24.), f(0.), atol=1e-12)
    np.testing.assert_allclose(f(25.5), f(1.5), atol=1e-12)
    h = 1e-4
    for d in (1, 2):
        def deriv(x):
            if d == 1:
                return (f(x + h) - f(x - h)) / (2 * h)
            return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
        np.testing.assert_allclose(deriv(24. - 1e-3), deriv(1e-3), rtol=1e-2, atol=1e-3)
    # the penalty is the integral of the squared second derivative; its
    # null space the constant
    xs = np.linspace(0, 24, 400001)
    f2 = np.gradient(np.gradient(f(xs), xs), xs)
    np.testing.assert_allclose(c @ spline.penalty @ c, np.trapezoid(f2**2, xs), rtol=1e-3)
    np.testing.assert_allclose(spline.penalty @ np.ones(m), 0, atol=1e-10)
    assert np.linalg.matrix_rank(spline.penalty) == m - 1


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_backfit_is_joint(data, family):
    fam, y = (sm.families.Gaussian(), 'y') if family == 'gaussian' else (sm.families.Poisson(), 'yp')
    fits = [GAM(terms=[cyclic_spline('hour', k=12, period=(0, 24), lam=1e-3), s('x', df=4)],
                family=fam, control=GAMControl(solver=solver, **TIGHT)).fit(data, data[y])
            for solver in ['backfit', 'joint']]
    np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-8)
    np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)


def test_predict_around_the_cycle(data):
    G = GAM(terms=[cyclic_spline('hour', period=(0, 24), lam=1e-3)]).fit(data, data['y'])
    term = G.terms_[0]
    np.testing.assert_allclose(term.predict(np.array([0., 24., 30.])),
                               term.predict(np.array([24., 0., 6.])), atol=1e-12)
    assert term.name == 'cyclic_spline(hour, period=(0, 24), lam=0.001)'


def test_plot(data):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    G = GAM(terms=[cyclic_spline('hour', period=(0, 24), lam=1e-3)]).fit(data, data['y'])
    axes = G.plot()
    assert len(axes) == 1


def test_errors(data):
    with pytest.raises(ValueError, match='outside the period'):
        GAM(terms=[cyclic_spline('hour', period=(0, 12), lam=1.)]).fit(data, data['y'])
    with pytest.raises(ValueError, match='give lam'):
        GAM(terms=[cyclic_spline('hour')]).fit(data, data['y'])
    with pytest.raises(ValueError, match='at least 4 knots'):
        CyclicCubicSpline([0., 1., 2.])


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_mgcv(Rinfo, data, family, method):
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    fam, y, Rfamily = ((sm.families.Gaussian(), 'y', 'gaussian()') if family == 'gaussian'
                       else (sm.families.Poisson(), 'yp', 'poisson()'))
    G = GLMM(terms=[cyclic_spline('hour', k=12, period=(0, 24))], family=fam, method=method,
            control=GAMControl(**TIGHT)).fit(data, data[y])
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
        rpy.r.assign('kn', G.terms_[0].spline_.knots)
    rpy.r(f"""m <- mgcv::gam({y} ~ s(hour, bs='cc', k=12), data=D, family={Rfamily},
                           method='{method}', knots=list(hour=kn),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    with Rinfo['np_cv_rules'].context():
        fitted = np.asarray(rpy.r('as.numeric(predict(m))'))
        edf = float(np.asarray(rpy.r('sum(m$edf)'))[0])
        score = float(np.asarray(rpy.r('as.numeric(m$gcv.ubre)'))[0])
    np.testing.assert_allclose(G.predict(data, prediction_type='link'), fitted, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(G.edf_.sum() + 1, edf, rtol=1e-7)
    # (no factor: the REML criterion's constant is mgcv's too)
    np.testing.assert_allclose(G.reml_score_, score, rtol=1e-9)
