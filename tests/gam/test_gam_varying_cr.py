"""
Factor-by cubic regression splines (`varying(cubic_regression_spline(x),
by=f)`, mgcv's `s(x, by=f, bs="cr")`): each level's curve zero off its
level and summing to 0 over the data (as mgcv's), backfitting against the
joint solve, the levels' order and unseen levels, and against mgcv under
REML and ML, with a smoothing parameter per level and pooled (mgcv's `id`).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, s, factor, varying, cubic_regression_spline

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 450
    x = rng.uniform(0, 1, n)
    f = rng.choice(['a', 'b', 'c'], n)
    curve = np.where(f == 'a', np.sin(2 * np.pi * x),
                     np.where(f == 'b', 2 * x**2, 0.5 * np.cos(4 * x)))
    D = pd.DataFrame({'x': x, 'f': f, 'z': rng.uniform(0, 1, n)})
    D['y'] = curve + (f == 'b') + np.sin(3 * D['z']) + 0.3 * rng.standard_normal(n)
    D['yp'] = rng.poisson(np.exp(0.3 + 0.5 * curve))
    return D


def test_curves(data):
    G = GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', lam=1e-3)]).fit(data, data['y'])
    term = G.terms_[1]
    fit = term.predict(data[['x', 'f']].to_numpy())
    # zero off its level; summing to 0 over all the rows' x, as mgcv's
    for level in ['a', 'b', 'c']:
        rows = (data['f'] == level).to_numpy()
        as_level = np.column_stack([data['x'], np.full(len(data), level, dtype=object)])
        np.testing.assert_allclose(term.predict(as_level).sum(), 0, atol=1e-10)
        other = np.column_stack([data['x'], np.where(rows, level, 'zz')])
        assert np.all(term.predict(other)[~rows] == 0)
    np.testing.assert_allclose(term.level_edf().sum(), G.edf_.iloc[1], rtol=1e-10)
    assert term.lam == [1e-3] * 3
    assert term.name == 'varying(cubic_regression_spline(x), by=f, lam=0.001)'


@pytest.mark.parametrize('pooled', [False, True])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_backfit_is_joint(data, family, pooled):
    fam, y = (sm.families.Gaussian(), 'y') if family == 'gaussian' else (sm.families.Poisson(), 'yp')
    lam = 1e-3 if pooled else [1e-4, 1e-3, 1e-2]
    fits = [GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', lam=lam, pooled_lam=pooled),
                       s('z', df=4)],
                family=fam, control=GAMControl(solver=solver, **TIGHT)).fit(data, data[y])
            for solver in ['backfit', 'joint']]
    np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-8)
    np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)


def test_categorical_order(data):
    D = data.assign(f=pd.Categorical(data['f'], categories=['c', 'a', 'b', 'd']))
    G = GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', lam=[1e-4, 1e-3, 1e-2])]).fit(D, D['y'])
    H = GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', lam=[1e-3, 1e-2, 1e-4])]).fit(data, data['y'])
    # the levels in the categories' order (those seen), the lams with them
    assert list(G.terms_[1].levels_) == ['c', 'a', 'b']
    np.testing.assert_allclose(G.predict(D), H.predict(data), rtol=1e-8)
    new = D.iloc[:5].assign(f=pd.Categorical(['d'] * 5, categories=['c', 'a', 'b', 'd']))
    np.testing.assert_allclose(G.partial_effect(1, new), 0)


def test_errors(data):
    with pytest.raises(ValueError, match='one per penalty of each level'):
        GAM(terms=[varying(cubic_regression_spline('x'), by='f', lam=[1., 2.])]).fit(data, data['y'])
    with pytest.raises(ValueError, match='give lam'):
        GAM(terms=[varying(cubic_regression_spline('x'), by='f')]).fit(data, data['y'])


@pytest.mark.parametrize('pooled', [False, True])
@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_mgcv(Rinfo, data, family, method, pooled):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    fam, y, Rfamily = ((sm.families.Gaussian(), 'y', 'gaussian()') if family == 'gaussian'
                       else (sm.families.Poisson(), 'yp', 'poisson()'))
    G = GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', pooled_lam=pooled)],
            family=fam, method=method, control=GAMControl(**TIGHT)).fit(data, data[y])
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
        rpy.r.assign('kn', G.terms_[1].inner_.spline_.knots)
    rpy.r('D$f <- factor(D$f)')
    id = ', id=1' if pooled else ''
    rpy.r(f"""m <- mgcv::gam({y} ~ f + s(x, by=f, bs='cr', k=10{id}), data=D, family={Rfamily},
                           method='{method}', knots=list(x=kn),
                           control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    with Rinfo['np_cv_rules'].context():
        fitted = np.asarray(rpy.r('as.numeric(predict(m))'))
        # mgcv's smoothing parameters on our scale: sp / S.scale
        rho_R = np.log(np.asarray(rpy.r('m$sp')) /
                       np.asarray(rpy.r('sapply(m$smooth, function(s) s$S.scale)'))[:1 if pooled else 3])
    # mgcv stops early where the criterion is flat (ML, a level whose curve
    # is near linear: its smoothing parameter large): our criterion is at
    # most its value at mgcv's, and the fits and the finite parameters close
    rho = np.log(G.smoothing_.lam_raw)
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [1], method=method)

        def value(rho):
            if C.n_params == C.n_rho:
                return float(C.value_and_grad(rho)[0])
            # (the Gaussian's dispersion, at its best)
            return minimize_scalar(lambda lp: float(C.value_and_grad(np.r_[rho, lp])[0]),
                                   bounds=(-10, 5), method='bounded').fun

        assert value(rho) <= value(rho_R) + 1e-7
    finite = rho_R < 10
    np.testing.assert_allclose(rho[finite], rho_R[finite], atol=0.1)
    tight = method == 'REML' and family == 'gaussian'
    np.testing.assert_allclose(G.predict(data, prediction_type='link'), fitted,
                               atol=1e-8 if tight else 1e-2)


def test_plot(data):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    G = GAM(terms=[factor('f'), varying(cubic_regression_spline('x'), by='f', lam=1e-3)],
            control=GAMControl(solver='joint')).fit(data, data['y'])
    axes = G.plot()
    assert len(axes) == 2
    assert len(axes[1].get_lines()) == 4          # a curve per level, and the rug
