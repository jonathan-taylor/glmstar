"""
Structured covariances of correlated random effects
(`correlated_random_effect(..., covariance=...)`): compound symmetry (nlme's
`pdCompSymm`) and diagonal (`pdDiag`). Their penalties against differences;
compound symmetry with a positive correlation is `re(g) + re([g, condition])`
and diagonal is independent `re()` terms, exactly; the REML and ML
gradients; and against nlme's `lme`.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GLMM, GAMControl, re, linear, factor, correlated_random_effect as cre
from glmnet.gam.penalties import CompoundSymmetryPenalty, DiagonalPenalty

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)
Q = 3
CONDITIONS = [f'c{k}' for k in range(Q)]


@pytest.fixture(scope='module')
def data():
    # 40 subjects, each measured 4 times in each of 3 conditions
    rng = np.random.default_rng(0)
    G, n_rep = 40, 4
    subject = np.repeat(np.arange(G), Q * n_rep)
    cond = np.tile(np.repeat(np.arange(Q), n_rep), G)
    D = pd.DataFrame({'subject': subject, 'cond': cond.astype(str),
                      'x': rng.uniform(0, 2, len(subject))})
    for k in range(Q):
        D[f'c{k}'] = (cond == k).astype(float)
    b, bc = rng.normal(0, 1.0, G), rng.normal(0, 0.6, (G, Q))
    slope = rng.normal(0, 0.5, G)
    D['y'] = (1 + 0.5 * cond + b[subject] + bc[subject, cond] + slope[subject] * D['x'] +
              rng.normal(0, 0.5, len(D)))
    return D


@pytest.mark.parametrize('Penalty', [CompoundSymmetryPenalty, DiagonalPenalty])
def test_penalty(Penalty):
    idx = np.arange(12).reshape((4, 3))
    p = Penalty(idx, ['a'] * (2 if Penalty is CompoundSymmetryPenalty else 3))
    theta = np.array([0.3, 0.8]) if Penalty is CompoundSymmetryPenalty else np.array([0.2, -0.5, 1.])
    Sigma = p.covariance(theta)
    np.testing.assert_allclose(p.parameters(Sigma), theta, rtol=1e-10)
    np.testing.assert_allclose(p.theta(np.linalg.inv(Sigma)), theta, rtol=1e-10)
    e = p.evaluate(theta)
    L = e.V
    T = np.kron(np.identity(4), L)
    S_full = np.kron(np.identity(4), np.linalg.inv(Sigma))
    np.testing.assert_allclose(e.logdet, np.linalg.slogdet(S_full)[1], rtol=1e-10)
    eps = 1e-6
    for k in range(len(theta)):
        d = np.zeros(len(theta))
        d[k] = eps
        dS = np.zeros((12, 12))
        e.derivatives[k].add_to(dS)
        fd = (np.kron(np.identity(4), p.precision(theta + d)) -
              np.kron(np.identity(4), p.precision(theta - d))) / (2 * eps)
        np.testing.assert_allclose(dS, T.T @ fd @ T, rtol=1e-5, atol=1e-7)
        fd = (p.evaluate(theta + d).logdet - p.evaluate(theta - d).logdet) / (2 * eps)
        np.testing.assert_allclose(e.dlogdet[k], fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_compound_symmetry_is_nested_re(data, method):
    # Sigma = s1 J + s2 I: a subject effect and a subject-by-condition effect
    pytest.importorskip('jax')
    G = GLMM(terms=[factor('cond'), cre('subject', slopes=CONDITIONS, intercept=False,
                                       covariance='compound_symmetry')],
            method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    G0 = GLMM(terms=[factor('cond'), re('subject'), re(['subject', 'cond'])], method=method,
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.smoothing_.converged
    S = G.terms_[1].covariance_.to_numpy()
    v = G0.variance_components_['variance']
    np.testing.assert_allclose(S[0, 0], v.iloc[0] + v.iloc[1], rtol=1e-6)
    np.testing.assert_allclose(S[0, 1], v.iloc[0], rtol=1e-6)
    np.testing.assert_allclose(G.predict(data), G0.predict(data), rtol=1e-8)
    np.testing.assert_allclose(G.reml_score_, G0.reml_score_, rtol=1e-10)
    assert G.terms_[1].name.endswith("covariance='compound_symmetry')")


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_diagonal_is_independent_re(data, method):
    pytest.importorskip('jax')
    G = GLMM(terms=[linear('x'), cre('subject', slopes=['x'], covariance='diagonal')],
            method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    G0 = GLMM(terms=[linear('x'), re('subject'), re('subject', by='x')], method=method,
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    S = G.terms_[1].covariance_.to_numpy()
    np.testing.assert_allclose(np.diag(S), G0.variance_components_['variance'].iloc[:2], rtol=1e-6)
    np.testing.assert_allclose(S[0, 1], 0, atol=1e-12)
    np.testing.assert_allclose(G.predict(data), G0.predict(data), rtol=1e-8)


@pytest.mark.parametrize('structure', ['compound_symmetry', 'diagonal'])
def test_gradient(data, structure):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    G = GLMM(terms=[factor('cond'), cre('subject', slopes=CONDITIONS, intercept=False,
                                       covariance=structure)],
            method='REML', control=GAMControl(**TIGHT)).fit(data, data['y'])
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [1], method='REML')
        x = C.start() + np.random.default_rng(0).uniform(-0.5, 0.5, C.n_params)
        value, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                       for e in np.identity(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


def test_errors():
    with pytest.raises(ValueError, match='covariance should be one of'):
        cre('g', slopes=['x'], covariance='AR1')


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_lme_compound_symmetry(Rinfo, data, method):
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("nlme", quietly=TRUE))')[0]:
        pytest.skip('requires the R package nlme')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data[['subject', 'cond', 'y']])
    rpy.r(f"""D$cond <- factor(D$cond); D$subject <- factor(D$subject)
              m <- nlme::lme(y ~ cond, random=list(subject=nlme::pdCompSymm(~ cond - 1)), data=D,
                             method='{method}', control=nlme::lmeControl(tolerance=1e-12,
                                                                         msMaxIter=500))
              V <- nlme::getVarCov(m)""")
    G = GLMM(terms=[factor('cond'), cre('subject', slopes=CONDITIONS, intercept=False,
                                       covariance='compound_symmetry')],
            method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    with Rinfo['np_cv_rules'].context():
        V = np.asarray(rpy.r('as.matrix(V)'))
        sigma = float(np.asarray(rpy.r('m$sigma'))[0])
        fitted = np.asarray(rpy.r('as.numeric(fitted(m))'))
    np.testing.assert_allclose(G.terms_[1].covariance_.to_numpy(), V, rtol=1e-4)
    np.testing.assert_allclose(np.sqrt(G.dispersion_), sigma, rtol=1e-5)
    np.testing.assert_allclose(G.predict(data), fitted, rtol=1e-6)
    if method == 'ML':
        np.testing.assert_allclose(-G.reml_score_, float(np.asarray(rpy.r('as.numeric(logLik(m))'))[0]),
                                   rtol=1e-7)
