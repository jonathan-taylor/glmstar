"""
Terms with several penalties (`pen()`, a block of columns with penalties
`sum_m lam_m c' S_m c / 2`, as mgcv's `paraPen`): the fit at fixed
smoothing parameters is that of their weighted sum; REML and ML for a term
whose penalties split its columns are those of the separate terms (in a
rotated basis, so that the penalties are dense on the term's range); the
gradient against differences; and against mgcv's `paraPen`.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, s, re, pen

TIGHT = dict(epsnr=1e-12, mxitnr=100)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    n = 400
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'g': rng.choice([f'g{i}' for i in range(12)], n),
                      'w': rng.uniform(0.5, 2, n)})
    X = rng.standard_normal((n, 6))
    for k in range(6):
        D[f'x{k}'] = X[:, k]
    b = np.linspace(-2, 2, n)
    for k in range(4):
        D[f'b{k}'] = b**(k + 1) / 2**k
    beta = np.array([0.8, -0.5, 0.3, 0.1, 0.05, -0.05])
    effects = dict(zip(sorted(D['g'].unique()), rng.normal(0, 0.6, 12)))
    eta = np.sin(2 * np.pi * D['a']) + X @ beta + D['g'].map(effects) + 0.3 * D['b1']
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yp'] = rng.poisson(np.exp(0.3 * eta))
    return D


COLS = [f'x{k}' for k in range(6)]
FAMILIES = {'gaussian': (sm.families.Gaussian(), 'y'),
            'poisson': (sm.families.Poisson(), 'yp')}


def _rotation(K, seed=0):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((K, K)))[0]


def _overlapping():
    # a first difference penalty and a ridge on the last columns: they do
    # not commute, and their sum has full rank
    D1 = np.diff(np.identity(6), axis=0)
    return [D1.T @ D1, np.diag([0., 0., 0., 1., 1., 1.])]


@pytest.mark.parametrize('solver', ['backfit', 'joint'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_fixed_is_the_sum(data, family, solver):
    fam, y = FAMILIES[family]
    S1, S2 = _overlapping()
    lam = [0.02, 0.005]
    fits = []
    for term in [pen(COLS, [S1, S2], lam=lam), pen(COLS, [lam[0] * S1 + lam[1] * S2], lam=[1.])]:
        G = GAM(terms=[s('a', lam=1e-5), term], family=fam,
                control=GAMControl(solver=solver, **TIGHT)).fit(data, data[y])
        fits.append(G)
    np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)
    np.testing.assert_allclose(fits[0].posterior_covariance_, fits[1].posterior_covariance_,
                               rtol=1e-6, atol=1e-12)


def test_one_identity_penalty_is_re(data):
    # pen() on a group's indicators with the identity penalty is re()
    groups = sorted(data['g'].unique())
    D = data.copy()
    names = [f'is_{g}' for g in groups]
    for g, name in zip(groups, names):
        D[name] = (D['g'] == g).astype(float)
    G_re = GAM(terms=[s('a'), re('g')], method='REML', control=GAMControl(**TIGHT)).fit(D, D['y'])
    G_pen = GAM(terms=[s('a'), pen(names, [np.identity(len(groups))])], method='REML',
                control=GAMControl(**TIGHT)).fit(D, D['y'])
    np.testing.assert_allclose(G_pen.predict(D), G_re.predict(D), rtol=1e-6)
    np.testing.assert_allclose(G_pen.smoothing_.lam, G_re.smoothing_.lam, rtol=1e-5)
    np.testing.assert_allclose(G_pen.reml_score_, G_re.reml_score_, rtol=1e-9)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_split_penalties_are_separate_terms(data, family, method):
    # one term on the columns [x0..x5] Q with the penalties Q' diag(1,1,1,0,0,0) Q
    # and Q' diag(0,0,0,1,1,1) Q is the two terms on x0..x2 and x3..x5,
    # each with the identity penalty
    fam, y = FAMILIES[family]
    Q = _rotation(6)
    D = data.copy()
    rotated = [f'q{k}' for k in range(6)]
    D[rotated] = D[COLS].to_numpy() @ Q
    E1, E2 = np.diag([1., 1, 1, 0, 0, 0]), np.diag([0., 0, 0, 1, 1, 1])
    control = GAMControl(**TIGHT)
    G_two = GAM(terms=[s('a'), pen(COLS[:3], [np.identity(3)]), pen(COLS[3:], [np.identity(3)])],
                family=fam, method=method, control=control).fit(D, D[y])
    G_one = GAM(terms=[s('a'), pen(rotated, [Q.T @ E1 @ Q, Q.T @ E2 @ Q])],
                family=fam, method=method, control=control).fit(D, D[y])
    np.testing.assert_allclose(G_one.smoothing_.lam, G_two.smoothing_.lam, rtol=1e-4)
    np.testing.assert_allclose(G_one.reml_score_, G_two.reml_score_, rtol=1e-9)
    np.testing.assert_allclose(G_one.predict(D), G_two.predict(D), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(G_one.edf_.iloc[1], G_two.edf_.iloc[1:].sum(), rtol=1e-5)
    assert G_one.smoothing_.converged
    assert list(G_one.smoothing_.terms) == ['s(a)', f'pen({", ".join(rotated)})[0]',
                                            f'pen({", ".join(rotated)})[1]']
    assert len(G_one.variance_components_) == 4


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_gradient(data, family, method):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    fam, y = FAMILIES[family]
    G = GAM(terms=[s('a'), pen(COLS, _overlapping()), re('g')], family=fam, method=method,
            control=GAMControl(**TIGHT), weight_id='w', response_id=y).fit(data, data[[y, 'w']])
    estimated = [j for j, t in enumerate(G.terms_) if t.penalty_matrices()]
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, estimated, method=method)
        assert C.n_rho == 4
        x = C.start() + np.random.default_rng(0).uniform(-1, 1, C.n_params)
        value, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                       for e in np.identity(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(G.smoothing_.gradient, 0, atol=1e-5)
    assert G.smoothing_.converged


def test_errors(data):
    with pytest.raises(ValueError, match='one lam per penalty'):
        GAM(terms=[pen(COLS, _overlapping(), lam=[1.])]).fit(data, data['y'])
    with pytest.raises(ValueError, match='shape'):
        GAM(terms=[pen(COLS, [np.identity(3)], lam=[1.])]).fit(data, data['y'])
    with pytest.raises(ValueError, match='give lam'):
        GAM(terms=[pen(COLS, _overlapping())]).fit(data, data['y'])


def test_cox_not_supported(data):
    pytest.importorskip('coxdev')
    from glmnet.cox import CoxFamily
    D = data.copy()
    D['time'] = np.exp(-0.3 * D['y']) * np.random.default_rng(1).exponential(1, D.shape[0])
    D['status'] = 1
    with pytest.raises(NotImplementedError, match='several penalties'):
        GAM(terms=[s('a'), pen(COLS, _overlapping())],
            family=CoxFamily(event_id='time', status_id='status'), method='REML').fit(D, D)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_mgcv_parapen(Rinfo, data, family, method):
    # mgcv: gam(y ~ s(a, bs="cr") + B, paraPen=list(B=list(S1, S2)))
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    fam, y = FAMILIES[family]
    Rfamily = 'gaussian()' if family == 'gaussian' else 'poisson()'
    D1 = np.diff(np.identity(4), axis=0)
    S1, S2 = D1.T @ D1, np.diag([0., 0., 1., 1.])
    bcols = [f'b{k}' for k in range(4)]
    G = GAM(terms=[s('a'), pen(bcols, [S1, S2])], family=fam, method=method,
            control=GAMControl(**TIGHT)).fit(data, data[y])
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data[['a', y] + bcols])
        rpy.r.assign('S1', S1)
        rpy.r.assign('S2', S2)
    ka = data['a'].nunique()
    rpy.r(f"""D$B <- as.matrix(D[, c('b0', 'b1', 'b2', 'b3')])
              m <- mgcv::gam({y} ~ s(a, bs='cr', k={ka}) + B, data=D, family={Rfamily},
                             paraPen=list(B=list(S1, S2)), method='{method}',
                             knots=list(a=sort(unique(D$a))),
                             control=mgcv::gam.control(epsilon=1e-12, maxit=200,
                                                       newton=list(conv.tol=1e-10)))""")
    with Rinfo['np_cv_rules'].context():
        fitted = np.asarray(rpy.r('as.numeric(predict(m))'))
        # mgcv's sp: the paraPen penalties' (as given), then the smooth's
        # (its penalty rescaled by S.scale)
        sp = np.asarray(rpy.r('as.numeric(m$sp)'))
        scale = float(np.asarray(rpy.r('m$smooth[[1]]$S.scale'))[0])
        edf = np.asarray(rpy.r('as.numeric(m$edf)'))
    ours = G.smoothing_.lam_raw            # s(a), then the penalties of pen()
    theirs = np.array([sp[2] / scale, sp[0], sp[1]])
    # a smoothing parameter at infinity (in effect) in both: the difference
    # penalty's here, whose null space (equal coefficients) the data favour
    infinite = (ours > 1e8) & (theirs > 1e8)
    # (the criterion is flat in the smoothing parameter of a term with
    # little effect: to a few parts in 1e3)
    np.testing.assert_allclose(ours[~infinite], theirs[~infinite], rtol=5e-3)
    np.testing.assert_allclose(G.predict(data, prediction_type='link'), fitted, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(G.edf_.iloc[1], edf[1:5].sum(), rtol=5e-3, atol=1e-5)
    assert G.smoothing_.converged
    if method == 'ML':
        # the ML criterion is mgcv's (REML's differs by a constant of the
        # parametrization of the unpenalized coefficients): ours is as low
        # (mgcv can stop short where it is flat, along a smoothing parameter
        # going to infinity)
        score = float(np.asarray(rpy.r('as.numeric(m$gcv.ubre)'))[0])
        assert G.reml_score_ <= score + 1e-9 * abs(score)
        np.testing.assert_allclose(G.reml_score_, score, rtol=1e-5)
