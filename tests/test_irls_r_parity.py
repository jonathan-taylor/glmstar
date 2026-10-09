"""
The IRLS GLMNet against R's glmnet with a family object (glmnet.path), with
both solved to a tight tolerance: the paths should agree to ~1e-8, with the
same lambda values and path length.
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
}

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
    np.testing.assert_allclose(G0.lambda_values_, R['lambda0'], rtol=1e-6)
    # user lambda values: the whole path is fit
    assert G.coefs_.shape[0] == R['lambda'].shape[0]
    np.testing.assert_allclose(G.coefs_, R['beta'], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(G.intercepts_, R['a0'], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(G.summary_['Fraction Deviance Explained'], R['dev_ratio'], atol=1e-7)
    for k, v in kw.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(getattr(G, k), v)


@pytest.mark.parametrize('fam_name', list(FAMILIES))
def test_irls_matches_R(Rinfo, data, fam_name):
    check(Rinfo, data, fam_name, 'plain')


@pytest.mark.parametrize('fam_name', ['logit', 'poisson'])
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
