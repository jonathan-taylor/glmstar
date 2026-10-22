"""
Correlated random effects (`correlated_random_effect()`, lme4's `(1 + v | g)`):
the precision's log-Cholesky penalty, backfitting against the joint solve,
the REML and ML gradients, against lme4's `lmer` (REML and ML) and `glmer`
(Laplace, our ML), and against two independent `re()` terms when the
covariance is diagonal.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import (GAM, GAMControl, linear, re, s,
                        correlated_random_effect as cre)
from glmnet.gam.penalties import CholeskyPenalty

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    # 18 subjects, 10 days each, a random intercept and slope, correlated
    rng = np.random.default_rng(0)
    G, T = 18, 10
    D = pd.DataFrame({'subject': np.repeat([f's{i:02d}' for i in range(G)], T),
                      'days': np.tile(np.arange(T, dtype=float), G),
                      'x': rng.uniform(0, 1, G * T)})
    Sigma = np.array([[25.**2, 0.4 * 25 * 6], [0.4 * 25 * 6, 6.**2]])
    b = rng.multivariate_normal([0, 0], Sigma, G)
    code = np.repeat(np.arange(G), T)
    D['y'] = (250 + 10 * D['days'] + 20 * np.sin(2 * np.pi * D['x']) + b[code, 0] +
              b[code, 1] * D['days'] + 25 * rng.standard_normal(G * T))
    eta = -1 + 0.2 * D['days'] + (b[code, 0] + b[code, 1] * D['days']) / 30
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return D


def test_cholesky_penalty():
    # the precision I (x) (L L')^{-1}, given in the coordinates u of b = L u
    # (per group): there the identity, its derivatives L' dS L
    idx = np.arange(12).reshape((4, 3))
    p = CholeskyPenalty(idx, ['a'] * 6)
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(6)
    B = p.precision(theta)
    np.testing.assert_allclose(p.theta(B), theta, rtol=1e-10)
    e = p.evaluate(theta)
    L = p.factor(theta)
    np.testing.assert_allclose(e.V, L)
    np.testing.assert_allclose(e.V_inverse @ L, np.identity(3), atol=1e-12)
    T = np.kron(np.identity(4), L)
    S = np.zeros((12, 12))
    e.matrix.add_to(S)
    np.testing.assert_allclose(S, T.T @ np.kron(np.identity(4), B) @ T, atol=1e-10)
    np.testing.assert_allclose(e.logdet, np.linalg.slogdet(np.kron(np.identity(4), B))[1], rtol=1e-10)
    np.testing.assert_allclose(e.logdet_V, np.linalg.slogdet(T)[1], rtol=1e-10)
    eps = 1e-6
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        dS = np.zeros((12, 12))
        e.derivatives[k].add_to(dS)
        fd = (np.kron(np.identity(4), p.precision(theta + d)) -
              np.kron(np.identity(4), p.precision(theta - d))) / (2 * eps)
        np.testing.assert_allclose(dS, T.T @ fd @ T, rtol=1e-5, atol=1e-7)
        fd = (p.evaluate(theta + d).logdet - p.evaluate(theta - d).logdet) / (2 * eps)
        np.testing.assert_allclose(e.dlogdet[k], fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_backfit_is_joint(data, family):
    fam, y = (sm.families.Gaussian(), 'y') if family == 'gaussian' else (sm.families.Binomial(), 'yb')
    B = np.array([[2e-1, 0.05], [0.05, 3.]])
    fits = [GAM(terms=[linear('days'), s('x', lam=1e-3), cre('subject', slopes=['days'], precision=B)],
                family=fam, control=GAMControl(solver=solver, **TIGHT)).fit(data, data[y])
            for solver in ['backfit', 'joint']]
    np.testing.assert_allclose(fits[0].predict(data), fits[1].predict(data), rtol=1e-8)
    np.testing.assert_allclose(fits[0].edf_, fits[1].edf_, rtol=1e-6)
    np.testing.assert_allclose(fits[0].terms_[2].effects_, fits[1].terms_[2].effects_, rtol=1e-6,
                               atol=1e-8)


def test_diagonal_is_two_re(data):
    # a diagonal precision: the independent intercepts and slopes of re()
    G1 = GAM(terms=[linear('days'), cre('subject', slopes=['days'],
                                        precision=np.diag([0.2, 3.]))],
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    G2 = GAM(terms=[linear('days'), re('subject', lam=0.2), re('subject', by='days', lam=3.)],
             control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(G1.predict(data), G2.predict(data), rtol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_gradient(data, family, method):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    fam, y = (sm.families.Gaussian(), 'y') if family == 'gaussian' else (sm.families.Binomial(), 'yb')
    G = GAM(terms=[linear('days'), s('x'), cre('subject', slopes=['days'])], family=fam,
            method=method, control=GAMControl(**TIGHT)).fit(data, data[y])
    assert G.smoothing_.converged
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [1, 2], method=method)
        assert C.n_rho == 4
        x = C.start() + np.random.default_rng(0).uniform(-0.5, 0.5, C.n_params)
        value, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                       for e in np.identity(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


def _R(Rinfo, D, package='lme4'):
    rpy = Rinfo['rpy']
    if not rpy.r(f'isTRUE(requireNamespace("{package}", quietly=TRUE))')[0]:
        pytest.skip(f'requires the R package {package}')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    return rpy


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_lmer(Rinfo, data, method):
    pytest.importorskip('jax')
    rpy = _R(Rinfo, data[['subject', 'days', 'y']])
    rpy.r(f"""m <- lme4::lmer(y ~ days + (1 + days | subject), data=D,
                             REML={'TRUE' if method == 'REML' else 'FALSE'},
                             control=lme4::lmerControl(optimizer='bobyqa',
                                                       optCtrl=list(rhoend=1e-12)))
              vc <- as.data.frame(lme4::VarCorr(m))""")
    G = GAM(terms=[linear('days'), cre('subject', slopes=['days'])], method=method,
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert G.smoothing_.converged
    Sigma = G.terms_[1].covariance_.to_numpy()
    sd = _get(Rinfo, 'vc$sdcor')            # intercept, slope, correlation, residual
    np.testing.assert_allclose(np.sqrt(np.diag(Sigma)), sd[:2], rtol=1e-5)
    np.testing.assert_allclose(Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1]), sd[2],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(G.dispersion_), sd[3], rtol=1e-6)
    np.testing.assert_allclose(G.predict(data), _get(Rinfo, 'as.numeric(fitted(m))'), rtol=1e-7)
    np.testing.assert_allclose(G.terms_[1].effects_.to_numpy(),
                               _get(Rinfo, 'as.matrix(lme4::ranef(m)$subject)'), rtol=1e-5,
                               atol=1e-6)
    if method == 'ML':
        # our criterion is the negative log-likelihood (lme4's at its optimum)
        np.testing.assert_allclose(G.reml_score_, -_get(Rinfo, 'as.numeric(logLik(m))')[0],
                                   rtol=1e-8)


def test_glmer(Rinfo, data):
    # binomial: our ML is the Laplace approximation, glmer's nAGQ=1 (with the
    # fixed effects integrated by Laplace too, glmer's nAGQ=0 and 1 differ)
    pytest.importorskip('jax')
    rpy = _R(Rinfo, data[['subject', 'days', 'yb']])
    rpy.r("""m <- lme4::glmer(yb ~ days + (1 + days | subject), data=D, family=binomial, nAGQ=0,
                              control=lme4::glmerControl(optimizer='bobyqa', tolPwrss=1e-12,
                                                         optCtrl=list(rhoend=1e-10)))
             vc <- as.data.frame(lme4::VarCorr(m))""")
    G = GAM(terms=[linear('days'), cre('subject', slopes=['days'])], family=sm.families.Binomial(),
            method='ML', control=GAMControl(**TIGHT)).fit(data, data['yb'])
    assert G.smoothing_.converged
    Sigma = G.terms_[1].covariance_.to_numpy()
    sd = _get(Rinfo, 'vc$sdcor')
    np.testing.assert_allclose(np.sqrt(np.diag(Sigma)), sd[:2], rtol=1e-4)
    np.testing.assert_allclose(G.predict(data, prediction_type='link'),
                               _get(Rinfo, 'as.numeric(predict(m))'), rtol=1e-5, atol=1e-6)
