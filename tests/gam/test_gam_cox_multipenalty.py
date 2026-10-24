"""
REML and ML for Cox GAMs with terms of several penalties, tensor products
and correlated random effects: a term whose penalties split its columns is
the separate terms, the gradient against differences, and mgcv's
`cox.ph` with `te(x, z, bs="cr")`.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')
pytest.importorskip('coxdev')

from glmnet.cox import CoxFamily
from glmnet.gam import (GAM, GLMM, GAMControl, s, penalized_columns, tensor_spline as te,
                        correlated_random_effect as cre)

TIGHT = GAMControl(epsnr=1e-12, mxitnr=500, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 300
    D = pd.DataFrame({'x': np.round(rng.uniform(0, 1, n), 2), 'z': rng.uniform(-1, 1, n),
                      'g': rng.choice([f'g{i}' for i in range(20)], n)})
    for k in range(6):
        D[f'c{k}'] = rng.standard_normal(n)
    # (a covariance well inside the positive definite: at a singular one, the
    # Cox gradient's difference of the information is not precise enough)
    b = rng.multivariate_normal([0, 0], [[0.8, 0.3], [0.3, 0.6]], 20)
    code = D['g'].map({f'g{i}': i for i in range(20)}).to_numpy()
    eta = (np.sin(2 * np.pi * D['x']) * D['z'] + 0.5 * D['c0'] - 0.3 * D['c4'] +
           b[code, 0] + b[code, 1] * D['z'])
    T = rng.exponential(1 / np.exp(eta))
    C = rng.exponential(2, n)
    D['time'] = np.minimum(T, C)
    D['status'] = (T <= C).astype(int)
    # a surface curved in both columns (sin(2 pi x) z is linear in z: the
    # tensor product's second smoothing parameter goes to infinity, where
    # the Cox gradient's precision, from differences of the information,
    # does not converge on every platform)
    eta2 = 1.2 * np.sin(2 * np.pi * D['x']) * np.cos(np.pi * D['z']) + b[code, 0] + b[code, 1] * D['z']
    T2 = rng.exponential(1 / np.exp(eta2))
    D['time2'] = np.minimum(T2, C)
    D['status2'] = (T2 <= C).astype(int)
    return D


def _family(event='time', status='status'):
    return CoxFamily(event_id=event, status_id=status, tie_breaking='breslow')


COLS = [f'c{k}' for k in range(6)]


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_split_penalties_are_separate_terms(data, method):
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((6, 6)))[0]
    D = data.copy()
    rotated = [f'q{k}' for k in range(6)]
    D[rotated] = D[COLS].to_numpy() @ Q
    E1, E2 = np.diag([1., 1, 1, 0, 0, 0]), np.diag([0., 0, 0, 1, 1, 1])
    G_two = GLMM(terms=[s('x'), penalized_columns(COLS[:3], [np.identity(3)]),
                       penalized_columns(COLS[3:], [np.identity(3)])],
                family=_family(), method=method, control=TIGHT).fit(D, D)
    G_one = GLMM(terms=[s('x'), penalized_columns(rotated, [Q.T @ E1 @ Q, Q.T @ E2 @ Q])],
                family=_family(), method=method, control=TIGHT).fit(D, D)
    assert G_one.smoothing_.converged and G_two.smoothing_.converged
    # (a smoothing parameter going to infinity, in both: where the
    # criterion is flat, only the finite are compared)
    finite = G_two.smoothing_.lam < 1e3
    np.testing.assert_allclose(G_one.smoothing_.lam[finite], G_two.smoothing_.lam[finite], rtol=1e-3)
    np.testing.assert_allclose(G_one.reml_score_, G_two.reml_score_, rtol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_gradient(data, method):
    from glmnet.gam.reml import CoxLaplaceCriterion
    G = GLMM(terms=[te('x', 'z'), cre('g', slopes=['z'])], family=_family('time2', 'status2'),
            method=method, control=TIGHT).fit(data, data)
    assert G.smoothing_.converged
    C = CoxLaplaceCriterion(G, [0, 1], method=method)
    assert C.n_rho == 5
    x = C.start() + np.random.default_rng(0).uniform(-0.5, 0.5, C.n_params)
    value, grad = C.value_and_grad(x)
    eps = 1e-4
    fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                   for e in np.identity(len(x))])
    # (the third derivatives are differences of the information: to 1e-5)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-4)


def test_mgcv_te(Rinfo, data):
    # mgcv's cox.ph with te(x, z, bs="cr") (its knots are ours), REML: both
    # smoothing parameters go to infinity (edf near 3, the null space x, z,
    # x z); mgcv stops sooner, and our criterion is as low
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    G = GLMM(terms=[te('x', 'z')], family=_family(), method='REML', control=TIGHT).fit(data, data)
    assert G.smoothing_.converged
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data[['x', 'z', 'time', 'status']])
    rpy.r("""m <- mgcv::gam(time ~ te(x, z, bs='cr'), family=mgcv::cox.ph(), weights=status,
                          data=D, method='REML',
                          control=mgcv::gam.control(epsilon=1e-12, maxit=200))""")
    eta = G.state_.link_parameter
    with Rinfo['np_cv_rules'].context():
        eta_R = np.asarray(rpy.r('m$linear.predictors'))
        edf_R = float(np.asarray(rpy.r('sum(m$edf)'))[0])
        score_R = float(np.asarray(rpy.r('as.numeric(m$gcv.ubre)'))[0])
    np.testing.assert_allclose(eta - eta.mean(), eta_R - eta_R.mean(), atol=3e-3)
    np.testing.assert_allclose(G.edf_.sum(), edf_R, atol=2e-2)
    assert G.reml_score_ <= score_R + 1e-8 * abs(score_R)
