"""
Overdispersion under REML and ML (`glmnet.gam.families`): quasi families
(the dispersion estimated with the smoothing parameters, as mgcv's
`quasipoisson()` and `quasibinomial()`) and the negative binomial (its
theta estimated, as mgcv's `nb()`, or given, its `negbin(theta)`). The
criterion's gradient against differences, and the fits against mgcv.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
jax = pytest.importorskip('jax')

from glmnet.gam import (GAM, GAMControl, s, categorical, QuasiPoisson, QuasiBinomial,
                        NegativeBinomial)
from glmnet.gam.reml import LaplaceCriterion

TIGHT = dict(epsnr=1e-12, mxitnr=100)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    n = 500
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n) * 40) / 40,
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    eta = (0.8 + np.sin(2 * np.pi * D['a']) + 0.2 * D['b']**2 +
           D['c'].map({'x': 0, 'y': .4, 'z': -.4}) + D['o'])
    mu = np.exp(eta)
    D['ynb'] = rng.negative_binomial(2.5, 2.5 / (2.5 + mu))
    # binomial proportions of 20 trials, overdispersed (a beta-binomial)
    p = rng.beta(4 / (1 + np.exp(-(eta - 1.5))) * 3, 3 * (1 - 1 / (1 + np.exp(-(eta - 1.5)))) * 4)
    D['trials'] = 20.
    D['pb'] = rng.binomial(20, p) / 20
    return D


def _R(Rinfo, D):
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    rpy.r('D$c <- factor(D$c)')
    return rpy


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))


def _mgcv(rpy, D, formula_y, Rfamily, method, weights=''):
    ka, kb = D['a'].nunique(), D['b'].nunique()
    rpy.r(f"""m <- mgcv::gam({formula_y} ~ s(a, bs='cr', k={ka}) + s(b, bs='cr', k={kb}) + c
                              + offset(o),
               data=D, family={Rfamily}, method='{method}'{weights},
               knots=list(a=sort(unique(D$a)), b=sort(unique(D$b))),
               control=mgcv::gam.control(epsilon=1e-12, maxit=200, newton=list(conv.tol=1e-10)))""")


def _fit(D, family, method, y):
    if y == 'pb':
        return GAM(terms=[s('a'), s('b'), categorical('c')], family=family, method=method,
                   control=GAMControl(**TIGHT), response_id='pb', weight_id='trials',
                   offset_id='o').fit(D, D[['pb', 'trials', 'o']])
    return GAM(terms=[s('a'), s('b'), categorical('c')], family=family, method=method,
               control=GAMControl(**TIGHT), response_id=y,
               offset_id='o').fit(D, D[[y, 'o']])


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['nb', 'quasipoisson'])
def test_gradient(data, family, method):
    # the gradient (in the smoothing parameters, the dispersion, theta)
    # against central differences, away from the optimum
    fam = NegativeBinomial() if family == 'nb' else QuasiPoisson()
    G = _fit(data, fam, method, 'ynb')
    assert G.smoothing_.converged
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [0, 1], method=method)
        assert C.n_params == 3
        x = C.start() + np.random.default_rng(0).uniform(-0.5, 0.5, C.n_params)
        _, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = [(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
              for e in np.identity(C.n_params)]
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_negative_binomial_mgcv(Rinfo, data, method):
    D = data
    rpy = _R(Rinfo, D)
    G = _fit(D, NegativeBinomial(), method, 'ynb')
    assert G.smoothing_.converged
    _mgcv(rpy, D, 'ynb', 'mgcv::nb()', method)
    theta_R = _get(Rinfo, 'm$family$getTheta(TRUE)')[0]
    np.testing.assert_allclose(G.negative_binomial_theta_, theta_R, rtol=1e-4)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-4, atol=1e-4)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_.values[:2], edf_R, rtol=1e-3, atol=1e-3)
    assert G.dispersion_ == 1.
    if method == 'ML':
        score_R = _get(Rinfo, 'm$gcv.ubre')[0]
        np.testing.assert_allclose(G.reml_score_, score_R, rtol=1e-7)
    # the null deviance at the estimated theta, and the AIC counting it
    np.testing.assert_allclose(G.null_deviance_, _get(Rinfo, 'm$null.deviance')[0], rtol=1e-6)
    assert np.isfinite(G.aic_)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_negative_binomial_given_theta(Rinfo, data, method):
    # a given theta: mgcv's negbin(theta)
    D = data
    rpy = _R(Rinfo, D)
    G = _fit(D, NegativeBinomial(theta=2.), method, 'ynb')
    assert G.smoothing_.converged and G.negative_binomial_theta_ == 2.
    _mgcv(rpy, D, 'ynb', 'mgcv::negbin(2)', method)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-4, atol=1e-4)
    if method == 'ML':
        np.testing.assert_allclose(G.reml_score_, _get(Rinfo, 'm$gcv.ubre')[0], rtol=1e-7)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['quasipoisson', 'quasibinomial'])
def test_quasi_mgcv(Rinfo, data, family, method):
    D = data
    rpy = _R(Rinfo, D)
    if family == 'quasipoisson':
        G = _fit(D, QuasiPoisson(), method, 'ynb')
        _mgcv(rpy, D, 'ynb', 'quasipoisson()', method)
    else:
        G = _fit(D, QuasiBinomial(), method, 'pb')
        _mgcv(rpy, D, 'pb', 'quasibinomial()', method, weights=', weights=trials')
    assert G.smoothing_.converged
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-4, atol=1e-4)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_.values[:2], edf_R, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G.dispersion_, _get(Rinfo, 'm$reml.scale')[0], rtol=1e-4)
    if method == 'ML':
        np.testing.assert_allclose(G.reml_score_, _get(Rinfo, 'm$gcv.ubre')[0], rtol=1e-7)
    # a quasi-likelihood: no log-likelihood or AIC
    assert np.isnan(G.loglike_) and np.isnan(G.aic_)


def test_quasi_fixed_smoothing(data):
    # with fixed smoothing parameters, Pearson's estimate (as R's glm)
    D = data
    G = GAM(terms=[s('a', lam=1e-3), s('b', lam=1e-2), categorical('c')], family=QuasiPoisson(),
            response_id='ynb', offset_id='o').fit(D, D[['ynb', 'o']])
    P = GAM(terms=[s('a', lam=1e-3), s('b', lam=1e-2), categorical('c')], family=sm.families.Poisson(),
            response_id='ynb', offset_id='o').fit(D, D[['ynb', 'o']])
    mu = np.asarray(P.state_.mu)
    np.testing.assert_allclose(np.asarray(G.state_.mu), mu, rtol=1e-10)
    pearson = np.sum((D['ynb'] - mu)**2 / mu) / P.df_resid_
    np.testing.assert_allclose(G.dispersion_, pearson, rtol=1e-8)
    assert P.dispersion_ == 1.


def test_theta_needs_reml(data):
    with pytest.raises(ValueError, match='estimated by method="REML"'):
        GAM(terms=[s('a', lam=1e-3)], family=NegativeBinomial(),
            response_id='ynb').fit(data, data[['ynb']])
    # fixed theta, fixed smoothing: as statsmodels' NegativeBinomial(alpha=1/theta)
    G = GAM(terms=[s('a', lam=1e-3)], family=NegativeBinomial(theta=2.),
            response_id='ynb').fit(data, data[['ynb']])
    H = GAM(terms=[s('a', lam=1e-3)], family=sm.families.NegativeBinomial(alpha=0.5),
            response_id='ynb').fit(data, data[['ynb']])
    np.testing.assert_allclose(G.predict(data), H.predict(data), rtol=1e-10)


def test_family_not_changed(data):
    family = NegativeBinomial()
    G = _fit(data, family, 'ML', 'ynb')
    assert family.estimate_theta and family.alpha == 1.
    assert G.negative_binomial_theta_ != 1.
    assert repr(family) == 'NegativeBinomial(theta=estimated)'
