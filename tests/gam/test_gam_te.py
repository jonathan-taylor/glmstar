"""
Tensor product smooths (`tensor_spline()`, mgcv's `te(..., bs="cr")`): the cubic
regression spline margins (their knots, basis and penalty), the products'
reproduction of the null space, backfitting against the joint solve,
prediction, the REML gradient, and against mgcv under REML and ML.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, s, factor, tensor_spline as te
from glmnet.gam.tensor import CubicRegressionSpline, place_knots

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 400
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(-1, 2, n),
                      'u': np.round(rng.uniform(0, 3, n), 1), 'c': rng.choice(list('ab'), n),
                      'w': rng.uniform(0.5, 2, n)})
    eta = np.sin(3 * D['x']) * D['z'] + 0.3 * D['u'] + 0.4 * (D['c'] == 'b')
    D['y'] = eta + 0.2 * rng.standard_normal(n)
    D['yp'] = rng.poisson(np.exp(0.5 * eta))
    return D


def test_cubic_regression_spline():
    rng = np.random.default_rng(1)
    knots = np.sort(rng.uniform(0, 1, 7))
    m = CubicRegressionSpline(knots)
    # the coefficients are the values at the knots
    np.testing.assert_allclose(m.basis(knots), np.identity(7), atol=1e-14)
    # a natural cubic spline: linear functions reproduced, inside and out
    x = np.linspace(-0.5, 1.5, 101)
    np.testing.assert_allclose(m.basis(x) @ (2 + 3 * knots), 2 + 3 * x, atol=1e-12)
    # the penalty is the integral of the squared second derivative
    c = rng.standard_normal(7)
    xs = np.linspace(knots[0], knots[-1], 200001)
    f2 = np.gradient(np.gradient(m.basis(xs) @ c, xs), xs)
    np.testing.assert_allclose(c @ m.penalty @ c, np.trapezoid(f2[100:-100]**2, xs[100:-100]),
                               rtol=1e-3)
    np.testing.assert_allclose(m.penalty @ (2 + 3 * knots), 0, atol=1e-12 * np.abs(m.penalty).max())


def test_place_knots():
    x = np.r_[0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
    np.testing.assert_allclose(place_knots(x, 4), [0, 3, 6, 9])
    np.testing.assert_allclose(place_knots(x, 5), [0, 2.25, 4.5, 6.75, 9])


def test_null_space(data):
    term = te('x', 'z').setup(data[['x', 'z']].to_numpy())
    B = term.basis()
    np.testing.assert_allclose(B.sum(1), 1, atol=1e-14)
    # 1, x, z and x z are reproduced, and not penalized
    np.testing.assert_allclose(B @ term._lin_coef, term._lin_cols, atol=1e-12)
    for S in term.penalty_matrices():
        np.testing.assert_allclose(S @ term._lin_coef, 0, atol=1e-12 * np.abs(S).max())
    assert len(term.penalty_matrices()) == 2


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
@pytest.mark.parametrize('modified', [True, False])
def test_backfit_is_joint(data, family, modified):
    fam, y = ((sm.families.Gaussian(), 'y') if family == 'gaussian'
              else (sm.families.Poisson(), 'yp'))
    terms = lambda: [te('x', 'z', lam=[1e-3, 1e-2]), s('u', lam=1e-3), factor('c')]
    fits = [GAM(terms=terms(), family=fam, weight_id='w', response_id=y,
                control=GAMControl(solver=solver, bf_modified=modified, **TIGHT)).fit(data, data[[y, 'w']])
            for solver in ['backfit', 'joint']]
    np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-7)
    np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)


def test_predict_and_three_columns(data):
    G = GAM(terms=[te('x', 'z', 'u', k=4, lam=[1e-2, 1e-2, 1e-2])],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    new = data.iloc[:7]
    np.testing.assert_allclose(G.predict(new), G.predict(data)[:7], rtol=1e-10)
    assert G.terms_[0].basis().shape[1] == 64
    mean, sd = G.predict(new, return_std=True)
    assert np.all(sd > 0)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_reml_gradient(data, method):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    G = GAM(terms=[te('x', 'z'), s('u')], family=sm.families.Poisson(), method=method,
            control=GAMControl(**TIGHT)).fit(data, data['yp'])
    assert G.smoothing_.converged
    np.testing.assert_allclose(G.smoothing_.gradient, 0, atol=1e-5)
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [0, 1], method=method)
        x = C.start() + np.random.default_rng(0).uniform(-1, 1, C.n_params)
        value, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                       for e in np.identity(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


def test_errors(data):
    with pytest.raises(ValueError, match='at least two columns'):
        te('x')
    with pytest.raises(ValueError, match='one lam per margin'):
        GAM(terms=[te('x', 'z', lam=[1.])]).fit(data, data['y'])
    with pytest.raises(ValueError, match='give lam'):
        GAM(terms=[te('x', 'z')]).fit(data, data['y'])


MGCV = {'gaussian': ('y', 'gaussian()', sm.families.Gaussian()),
        'poisson': ('yp', 'poisson()', sm.families.Poisson())}


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(MGCV))
def test_mgcv(Rinfo, data, family, method):
    # mgcv's te(x, z, bs="cr") with a smooth and a factor; its knots are ours
    # (place.knots); its sp are on the scale of its reparametrization, so
    # the fits, edf and criteria are compared
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    y, Rfamily, fam = MGCV[family]
    G = GAM(terms=[te('x', 'z', k=(5, 6)), s('u'), factor('c')], family=fam, method=method,
            control=GAMControl(**TIGHT)).fit(data, data[y])
    assert G.smoothing_.converged
    ku = data['u'].nunique()
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
    rpy.r(f"""m <- mgcv::gam({y} ~ te(x, z, bs='cr', k=c(5, 6)) + s(u, bs='cr', k={ku}) + c,
                           data=D, family={Rfamily}, method='{method}',
                           knots=list(u=sort(unique(D$u))),
                           control=mgcv::gam.control(epsilon=1e-12, maxit=200,
                                                     newton=list(conv.tol=1e-10)))""")
    with Rinfo['np_cv_rules'].context():
        fitted = np.asarray(rpy.r('as.numeric(predict(m))'))
        edf = np.asarray(rpy.r('as.numeric(m$edf)'))
        score = float(np.asarray(rpy.r('as.numeric(m$gcv.ubre)'))[0])
        first = int(np.asarray(rpy.r('m$smooth[[1]]$first.para'))[0]) - 1
        last = int(np.asarray(rpy.r('m$smooth[[1]]$last.para'))[0])
    # mgcv stops sooner where the criterion is flat (ours is as low): the
    # fits to about 1e-4
    np.testing.assert_allclose(G.predict(data, prediction_type='link'), fitted, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G.edf_.iloc[0], edf[first:last].sum(), rtol=1e-3)
    if method == 'ML':
        # (REML's criterion differs from mgcv's by a constant of the
        # parametrization of the factor's effects). A smoothing parameter
        # going to infinity stops at our bound (25 from its start, on the log
        # scale), mgcv's further: theirs can be lower, by about 1e-9
        assert G.reml_score_ <= score + 1e-8 * abs(score)
        np.testing.assert_allclose(G.reml_score_, score, rtol=1e-7)


def test_plot(data):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from glmnet.gam import plot_term
    from glmnet.gam.plot import plottable
    G = GAM(terms=[te('x', 'z', lam=[1e-3, 1e-2]), te('x', 'z', 'u', k=4, lam=[1e-2] * 3),
                   s('u', lam=1e-3)]).fit(data, data['y'])
    assert [plottable(t) for t in G.terms_] == [True, False, True]
    ax = plot_term(G, 0)
    assert ax.get_xlabel() == 'x' and ax.get_ylabel() == 'z'
    # the surface is the term's fit on the grid (filled contours)
    assert len(ax.collections) > 0
    with pytest.raises(ValueError, match='more than two columns'):
        plot_term(G, 1)
    axes = G.plot()
    assert len(axes) == 2
