"""
The IRLS GLMNet against R's glmnet with a family object (glmnet.path), with
both solved to a tight tolerance: the paths should agree to ~1e-8, with the
same lambda values and path length.

The Tweedie family is R's `statmod::tweedie`; it is skipped if statmod is not
installed.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl

N, P = 200, 10
TIGHT_R = 'thresh=1e-14, control=list(epsnr=1e-12, mxitnr=100)'
TIGHT = dict(thresh=1e-14, epsnr=1e-12, mxitnr=100)

FAMILIES = {
    'gaussian': (sm.families.Gaussian(), 'gaussian()'),
    'logit': (sm.families.Binomial(), 'binomial()'),
    'probit': (sm.families.Binomial(link=sm.families.links.Probit()), 'binomial(link="probit")'),
    'poisson': (sm.families.Poisson(), 'poisson()'),
    'gamma_log': (sm.families.Gamma(link=sm.families.links.Log()), 'Gamma(link="log")'),
    'cloglog': (sm.families.Binomial(link=sm.families.links.CLogLog()), 'binomial(link="cloglog")'),
    'cauchit': (sm.families.Binomial(link=sm.families.links.Cauchy()), 'binomial(link="cauchit")'),
    # known theta: variance mu + mu^2 / theta = mu + alpha mu^2
    'negbin': (sm.families.NegativeBinomial(alpha=1 / 3.), 'MASS::negative.binomial(theta=3)'),
    'invgauss_log': (sm.families.InverseGaussian(link=sm.families.links.Log()),
                     'inverse.gaussian(link="log")'),
    'invgauss': (sm.families.InverseGaussian(), 'inverse.gaussian()'),
    'gamma_identity': (sm.families.Gamma(link=sm.families.links.Identity()), 'Gamma(link="identity")'),
    'gamma_inverse': (sm.families.Gamma(), 'Gamma()'),
    'gaussian_log': (sm.families.Gaussian(link=sm.families.links.Log()), 'gaussian(link="log")'),
    'tweedie': (sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
                'statmod::tweedie(var.power=1.5, link.power=0)'),
}

# looser tolerances, by (family, option)
LAMBDA0_RTOL = {
    # with an offset R's null intercept for lambda_max is glm()'s, with
    # glm.control(epsilon=1e-8): Fisher scoring for the cloglog link converges
    # linearly and stops 5.6e-6 short (-0.01903332 vs -0.01903891), so R's
    # lambda values are 2.5e-6 off; glmstar's agree with R's when glm is run
    # to epsilon=1e-14
    ('cloglog', 'offset'): 1e-5,
}
COEF_ATOL = {
    # without an intercept the objective is flat at one lambda (index 44): the
    # coefficients differ by 1.2e-6 with penalized objectives equal to 1e-10,
    # and both R's and glmstar's move by 5e-7 when solved still tighter
    ('negbin', 'no_intercept'): 5e-6,
}

# R packages a family needs beyond glmnet (and MASS, a recommended package)
R_PACKAGES = {'tweedie': 'statmod'}

PF = np.r_[0.5, 2., 1., 3., np.ones(P - 4)]
OPTIONS = {
    'plain': ({}, ''),
    'penalty_factor': (dict(penalty_factor=PF), 'penalty.factor=pf'),
    'penalty_factor_zero': (dict(penalty_factor=np.r_[0., PF[1:]]), 'penalty.factor=c(0, pf[-1])'),
    'penalty_factor_inf': (dict(penalty_factor=np.r_[PF[:3], np.inf, PF[4:]]),
                           'penalty.factor=c(pf[1:3], Inf, pf[5:10])'),
    'alpha': (dict(alpha=0.4), 'alpha=0.4'),
    'limits': (dict(upper_limits=np.r_[0.1, np.full(P - 1, np.inf)],
                    lower_limits=np.r_[-np.inf, -0.05, np.full(P - 2, -np.inf)]),
               'upper.limits=c(0.1, rep(Inf, 9)), lower.limits=c(-Inf, -0.05, rep(-Inf, 8))'),
    'offset': (dict(offset_id='o'), 'offset=o'),
    'no_intercept': (dict(fit_intercept=False), 'intercept=FALSE'),
    'unstandardized': (dict(standardize=False), 'standardize=FALSE'),
    'exclude': (dict(exclude=[3, 5]), 'exclude=c(4, 6)'),
}


def tweedie_sample(rng, mu, p, phi):
    # compound Poisson-gamma, 1 < p < 2: Poisson(lam) many Gamma(shape, scale)
    lam = mu**(2 - p) / (phi * (2 - p))
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu**(p - 1)
    n = rng.poisson(lam)
    return np.where(n > 0, rng.gamma(np.maximum(n, 1) * shape, scale), 0.)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, P)) * rng.uniform(0.5, 3, P)
    eta = X[:, :3] @ np.array([0.3, -0.2, 0.15])
    prob = 1 / (1 + np.exp(-eta))
    D = pd.DataFrame({'gaussian': eta + rng.standard_normal(N),
                      'logit': rng.binomial(1, prob).astype(float),
                      'probit': rng.binomial(1, prob).astype(float),
                      'poisson': rng.poisson(np.exp(0.5 * eta)).astype(float),
                      'gamma_log': rng.gamma(2., np.exp(0.3 * eta) / 2),
                      'w': rng.uniform(0.5, 2, N),
                      'o': rng.normal(0, 0.3, N)})
    # drawn after the above, which are unchanged by adding them
    D['cloglog'] = rng.binomial(1, 1 - np.exp(-np.exp(eta))).astype(float)
    D['cauchit'] = rng.binomial(1, 0.5 + np.arctan(eta) / np.pi).astype(float)
    # numpy's (n, p): n = theta, p = theta / (theta + mu)
    D['negbin'] = rng.negative_binomial(3, 3 / (3 + np.exp(1 + 0.5 * eta))).astype(float)
    D['invgauss_log'] = rng.wald(np.exp(0.3 * eta), 4.)
    # canonical link 1 / mu^2 = eta: keep mu bounded and positive
    D['invgauss'] = rng.wald(1 / np.sqrt(1 + 0.1 * eta), 4.)
    D['gamma_identity'] = rng.gamma(4., (3 + 0.3 * eta) / 4)
    D['gamma_inverse'] = rng.gamma(4., 1 / (1 + 0.1 * eta) / 4)
    D['gaussian_log'] = np.exp(1 + 0.3 * eta) + 0.5 * rng.standard_normal(N)
    D['tweedie'] = tweedie_sample(rng, np.exp(0.5 * eta), 1.5, 1.)
    return X, D


def fit_R(Rinfo, X, D, fam_name, args):
    rpy = Rinfo['rpy']
    fam_R = FAMILIES[fam_name][1]
    extra = f', {args}' if args else ''
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('X', X)
        rpy.r.assign('yv', D[fam_name].values)
        rpy.r.assign('wv', D['w'].values)
        rpy.r.assign('ov', D['o'].values)
        rpy.r.assign('pfv', PF)
        rpy.r('y = as.vector(yv); w = as.vector(wv); o = as.vector(ov); pf = as.vector(pfv)')
        rpy.r('suppressMessages(library(glmnet))')
        # the default lambda values and path length, then a tight fit at them
        rpy.r(f'fit0 = suppressWarnings(glmnet(X, y, family={fam_R}, weights=w{extra}))')
        rpy.r(f'fit = suppressWarnings(glmnet(X, y, family={fam_R}, weights=w, lambda=fit0$lambda{extra}, '
              f'{TIGHT_R}))')
        return {'lambda0': np.asarray(rpy.r('fit0$lambda')),
                'lambda': np.asarray(rpy.r('fit$lambda')),
                'beta': np.asarray(rpy.r('as.matrix(fit$beta)')).T,
                'a0': np.asarray(rpy.r('fit$a0')),
                'dev_ratio': np.asarray(rpy.r('fit$dev.ratio'))}


def check(Rinfo, data, fam_name, option):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    pkg = R_PACKAGES.get(fam_name)
    if pkg:
        with Rinfo['np_cv_rules'].context():
            has_pkg = bool(np.ravel(Rinfo['rpy'].r(f'isTRUE(requireNamespace("{pkg}", quietly=TRUE))'))[0])
        if not has_pkg:
            pytest.skip(f'requires the R package {pkg}')
    X, D = data
    kw, args = OPTIONS[option]
    R = fit_R(Rinfo, X, D, fam_name, args)
    fam = FAMILIES[fam_name][0]

    def kwargs():
        # copies: the arrays must not be changed by fitting
        return {k: (np.copy(v) if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
                for k, v in kw.items()}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # default path: same lambda values and length as R's
        G0 = GLMNet(family=fam, response_id=fam_name, weight_id='w', **kwargs()).fit(X, D)
        G = GLMNet(family=fam, response_id=fam_name, weight_id='w', lambda_values=R['lambda'],
                   control=GLMNetControl(**TIGHT), **kwargs()).fit(X, D)

    assert G0.lambda_values_.shape == R['lambda0'].shape
    np.testing.assert_allclose(G0.lambda_values_, R['lambda0'],
                               rtol=LAMBDA0_RTOL.get((fam_name, option), 1e-6))
    # user lambda values: the whole path is fit
    assert G.coefs_.shape[0] == R['lambda'].shape[0]
    np.testing.assert_allclose(G.coefs_, R['beta'], atol=COEF_ATOL.get((fam_name, option), 1e-6), rtol=1e-6)
    np.testing.assert_allclose(G.intercepts_, R['a0'], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(G.summary_['Fraction Deviance Explained'], R['dev_ratio'], atol=1e-7)
    for k, v in kw.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(getattr(G, k), v)


@pytest.mark.parametrize('fam_name', list(FAMILIES))
def test_irls_matches_R(Rinfo, data, fam_name):
    check(Rinfo, data, fam_name, 'plain')


OPTION_FAMILIES = ['logit', 'poisson', 'cloglog', 'negbin']


@pytest.mark.parametrize('fam_name', OPTION_FAMILIES)
@pytest.mark.parametrize('option', [o for o in OPTIONS if o != 'plain'])
def test_irls_options_match_R(Rinfo, data, fam_name, option):
    check(Rinfo, data, fam_name, option)


def test_penalty_factor_not_modified(data):
    X, D = data
    pf = np.r_[PF[:3], np.inf, PF[4:]]
    pf_copy = pf.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        GLMNet(family=sm.families.Binomial(), response_id='logit', penalty_factor=pf).fit(X, D)
    np.testing.assert_array_equal(pf, pf_copy)
