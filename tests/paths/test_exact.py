import numpy as np
import pandas as pd
import pytest

from glmnet.paths import GaussNet, LogNet, FishNet, CoxNet
from glmnet.cox import CoxFamily

rng = np.random.default_rng(0)

def get_data(family, n, p, use_weights, use_offset):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1, -0.5, 0.25]
    eta = X @ beta
    if family == 'gaussian':
        D = pd.DataFrame({'Y': eta + rng.standard_normal(n)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 2))})
    else:
        T = rng.exponential(np.exp(-eta))
        C = rng.exponential(1.5, size=n)
        D = pd.DataFrame({'stop': np.round(np.minimum(T, C), 1) + 0.2,
                          'status': (T <= C).astype(int)})
    col_args = {}
    if family != 'cox':
        col_args['response_id'] = 'Y'
    if use_weights:
        D['weight'] = rng.uniform(0.5, 2, n)
        col_args['weight_id'] = 'weight'
    if use_offset:
        D['offset'] = 0.3 * rng.standard_normal(n)
        col_args['offset_id'] = 'offset'
    return X, D, col_args

def get_estimator(family, col_args):
    if family == 'gaussian':
        return GaussNet(**col_args)
    if family == 'binomial':
        return LogNet(**col_args)
    if family == 'poisson':
        return FishNet(**col_args)
    return CoxNet(family=CoxFamily(tie_breaking='breslow', event_id='stop',
                                   status_id='status'), **col_args)

def get_R_exact(Rinfo, family, X, D, s):
    """R's coef(glmnet(...), s=s, exact=TRUE, ...), shape (len(s), p) and intercepts."""
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    with np_cv_rules.context():
        rpy.r('rm(list=intersect(c("W", "O"), ls()))')
        rpy.r.assign('X', X)
        rpy.r.assign('s', s)
        if family == 'cox':
            rpy.r.assign('stop', D['stop'].values)
            rpy.r.assign('status', D['status'].values.astype(float))
            rpy.r('Y = Surv(stop, status)')
            extra = ', cox.ties="breslow"'
        else:
            rpy.r.assign('Y', D['Y'].values.astype(float))
            rpy.r('Y = as.vector(Y)')
            extra = ''
        args = ''
        if 'weight' in D.columns:
            rpy.r.assign('W', D['weight'].values)
            args += ', weights=as.vector(W)'
        if 'offset' in D.columns:
            rpy.r.assign('O', D['offset'].values)
            args += ', offset=as.vector(O)'
        rpy.r(f'''
suppressMessages({{library(glmnet); library(survival)}})
G = glmnet(X, Y, family="{family}"{extra}{args})
B = as.matrix(coef(G, s=as.vector(s), exact=TRUE, x=X, y=Y{extra}{args}))
''')
        B = np.asarray(rpy.r('B'))
    if family == 'cox':
        return B.T, np.zeros(B.shape[1])
    return B[1:].T, B[0]

@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_exact_coefs(Rinfo, family, use_weights, use_offset, n=150, p=8):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, col_args = get_data(family, n, p, use_weights, use_offset)
    L = get_estimator(family, col_args).fit(X, D)
    lam = L.lambda_values_
    # between path values, on a path value, and below the end of the path
    s = np.array([0.5 * (lam[3] + lam[4]), lam[10], 0.7 * lam[20] + 0.3 * lam[21], 0.5 * lam[-1]])

    coefs, intercepts = L.exact_coefs(X, D, s)
    R_coefs, R_intercepts = get_R_exact(Rinfo, family, X, D, s)

    assert coefs.shape == (s.shape[0], p)
    # R's extrapolated first lambda can differ from ours in the last bit; the
    # refit paths then differ within the solver's convergence tolerance
    assert np.allclose(coefs, R_coefs, rtol=1e-5, atol=1e-5)
    assert np.allclose(intercepts, R_intercepts, rtol=1e-5, atol=1e-5)

    # exact and interpolated coefficients agree on path values, up to the
    # different warm starts of the refit path
    interp_coefs, _ = L.interpolate_coefs(s)
    assert np.allclose(coefs[1], interp_coefs[1], rtol=1e-5, atol=1e-5)
    # but not in general between them (the gaussian lasso path is piecewise
    # linear in lambda, so interpolation there is often already exact)
    if family != 'gaussian':
        assert not np.allclose(coefs[0], interp_coefs[0])

def test_refit_path(n=100, p=5):

    X, D, col_args = get_data('gaussian', n, p, False, False)
    L = GaussNet(**col_args).fit(X, D)
    lam = L.lambda_values_.copy()
    s = 0.5 * (lam[5] + lam[6])

    R = L.refit_path(X, D, s)
    assert np.allclose(L.lambda_values_, lam)       # original unchanged
    assert L.lambda_values is None
    assert s in R.lambda_values_
    assert np.all(np.diff(R.lambda_values_) < 0)

    # the refit path agrees with the original at the original lambda values
    k = np.searchsorted(-R.lambda_values_, -lam[:20])
    assert np.allclose(R.coefs_[k], L.coefs_[:20], atol=1e-6)

    # a scalar lambda gives 1d coefficients
    coefs, intercept = L.exact_coefs(X, D, s)
    assert coefs.shape == (p,) and np.ndim(intercept) == 0

    # values already on the path need no new lambda values
    assert np.allclose(L.refit_path(X, D, lam[3]).lambda_values_, lam)

    with pytest.raises(ValueError, match='non-negative'):
        L.refit_path(X, D, -1.)
