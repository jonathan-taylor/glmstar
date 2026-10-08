import numpy as np
import pandas as pd
import pytest

from glmnet.paths import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet, CoxNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.cox import CoxFamily

N, P, PMAX = 200, 30, 5

def get_data(family):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((N, P))
    eta = X[:, :10] @ np.linspace(1, 0.2, 10)
    if family == 'gaussian':
        D = pd.DataFrame({'Y': eta + rng.standard_normal(N)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 4))})
    elif family == 'multinomial':
        logits = np.column_stack([eta, -eta, np.zeros(N)])
        prob = np.exp(logits) / np.exp(logits).sum(1)[:, None]
        D = pd.DataFrame({'Y': [rng.choice(3, p=p_) for p_ in prob]})
    elif family == 'mgaussian':
        D = pd.DataFrame({'Y1': eta + rng.standard_normal(N),
                          'Y2': -eta + rng.standard_normal(N)})
    else:
        T = rng.exponential(np.exp(-eta / 2))
        C = rng.exponential(1.5, size=N)
        D = pd.DataFrame({'stop': np.minimum(T, C),
                          'status': (T <= C).astype(int)})
    return X, D

def get_estimator(family, **args):
    args['control'] = FastNetControl(thresh=1e-14)
    if family == 'gaussian':
        return GaussNet(response_id='Y', **args)
    if family == 'binomial':
        return LogNet(response_id='Y', **args)
    if family == 'poisson':
        return FishNet(response_id='Y', **args)
    if family == 'multinomial':
        return MultiClassNet(response_id='Y', **args)
    if family == 'mgaussian':
        return MultiGaussNet(response_id=['Y1', 'Y2'], **args)
    return CoxNet(family=CoxFamily(tie_breaking='breslow', event_id='stop',
                                   status_id='status'), **args)

def R_fit(Rinfo, family, X, D, lambda_values, pmax):
    """Lambdas and coefficients of R's fit with this pmax (and its warning)."""
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        rpy.r('suppressMessages({library(glmnet); library(survival)})')
        rpy.r.assign('X', X)
        rpy.r.assign('lam', lambda_values)
        if family == 'cox':
            rpy.r.assign('st', D['stop'].values)
            rpy.r.assign('d', D['status'].values.astype(float))
            rpy.r('Y = Surv(as.vector(st), as.vector(d))')
        elif family == 'mgaussian':
            rpy.r.assign('Yv', D[['Y1', 'Y2']].values)
            rpy.r('Y = as.matrix(Yv)')
        elif family == 'multinomial':
            rpy.r.assign('Yv', D['Y'].values.astype(float))
            rpy.r('Y = factor(as.vector(Yv))')
        else:
            rpy.r.assign('Yv', D['Y'].values.astype(float))
            rpy.r('Y = as.vector(Yv)')
        rpy.r(f'''
        W = NULL
        G = withCallingHandlers(
              glmnet(X, Y, family="{family}", lambda=as.vector(lam),
                     control=list(thresh=1e-14, pmax={pmax})),
              warning=function(w) {{W <<- conditionMessage(w); invokeRestart("muffleWarning")}})
        B = if (is.list(G$beta)) sapply(G$beta, as.matrix, simplify="array") else as.matrix(G$beta)
        ''')
        lam_R = np.asarray(rpy.r('G$lambda'))
        B = np.asarray(rpy.r('B'))
        W = rpy.r('W')
        W = None if W is rpy.NULL or len(W) == 0 else str(W[0])
    # R's beta is (p, nlambda) or (p, nlambda, K): put lambda first
    B = np.moveaxis(B, 1, 0)
    return lam_R, B, W

@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'multinomial',
                                    'mgaussian', 'cox'])
def test_pmax_matches_R(Rinfo, family):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(family)
    full = get_estimator(family, nlambda=30).fit(X, D)

    with pytest.warns(UserWarning, match=f'exceeds pmax={PMAX}'):
        L = get_estimator(family,
                          lambda_values=full.lambda_values_,
                          pmax=PMAX).fit(X, D)
    lam_R, B_R, W_R = R_fit(Rinfo, family, X, D, full.lambda_values_, PMAX)

    assert W_R is not None and f'exceeds pmax={PMAX}' in W_R
    # the path stops at the same lambda as R's
    np.testing.assert_allclose(L.lambda_values_, lam_R)
    assert L.coefs_.shape[0] == lam_R.shape[0] < full.lambda_values_.shape[0]
    # and agrees with the fit without pmax up to there
    np.testing.assert_allclose(L.coefs_, full.coefs_[:L.coefs_.shape[0]],
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(L.coefs_, B_R, rtol=1e-4, atol=1e-6)
    # never more than pmax variables
    coefs = L.coefs_ if L.coefs_.ndim == 2 else np.abs(L.coefs_).sum(-1)
    assert np.all((coefs != 0).sum(1) <= PMAX)

def test_pmax_default():
    X, D = get_data('gaussian')
    L = get_estimator('gaussian', nlambda=10).fit(X, D)
    assert L._args['nx'] == P
    # with these data the path passes the default pmax=26 before df_max=3
    # stops it, and R warns in the same way
    with pytest.warns(UserWarning, match='exceeds pmax=26'):
        L = get_estimator('gaussian', nlambda=10, df_max=3).fit(X, D)
    assert L._args['nx'] == min(2 * 3 + 20, P)
    L = get_estimator('gaussian', nlambda=10, pmax=P).fit(X, D)
    assert L._args['nx'] == P

@pytest.mark.parametrize('pmax', [0, -1, 2.5])
def test_pmax_invalid(pmax):
    X, D = get_data('gaussian')
    with pytest.raises(ValueError, match='pmax'):
        get_estimator('gaussian', nlambda=10, pmax=pmax).fit(X, D)
