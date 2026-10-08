import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl
from glmnet.paths import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet, CoxNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.cox import CoxFamily

N, P = 150, 8

def get_data(family):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, P))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family in ['gaussian', 'irls']:
        D = pd.DataFrame({'Y': eta + rng.standard_normal(N)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 2))})
    elif family.startswith('multinomial'):
        logits = np.column_stack([eta, -eta, np.zeros(N)])
        prob = np.exp(logits) / np.exp(logits).sum(1)[:, None]
        D = pd.DataFrame({'Y': [rng.choice(3, p=p_) for p_ in prob]})
    elif family == 'mgaussian':
        D = pd.DataFrame({'Y1': eta + rng.standard_normal(N),
                          'Y2': -eta + rng.standard_normal(N)})
    else:
        T = rng.exponential(np.exp(-eta))
        C = rng.exponential(1.5, size=N)
        D = pd.DataFrame({'stop': np.minimum(T, C),
                          'status': (T <= C).astype(int)})
    return X, D

def get_estimator(family):
    control = FastNetControl(thresh=1e-14)
    if family == 'gaussian':
        return GaussNet(response_id='Y', control=control)
    if family == 'irls':
        return GLMNet(family=sm.families.Gaussian(), response_id='Y',
                      control=GLMNetControl(thresh=1e-14))
    if family == 'binomial':
        return LogNet(response_id='Y', control=control)
    if family == 'poisson':
        return FishNet(response_id='Y', control=control)
    if family == 'multinomial':
        return MultiClassNet(response_id='Y', control=control)
    if family == 'multinomial_grouped':
        return MultiClassNet(response_id='Y', grouped=True, control=control)
    if family == 'mgaussian':
        return MultiGaussNet(response_id=['Y1', 'Y2'], control=control)
    return CoxNet(family=CoxFamily(tie_breaking='breslow', event_id='stop',
                                   status_id='status'), control=control)

def R_nonzero(Rinfo, family, X, D, lambda_values, s=None):
    """R's predict(type="nonzero") as 0-based feature indices."""
    rpy = Rinfo['rpy']
    R_family = {'irls': 'gaussian',
                'multinomial_grouped': 'multinomial'}.get(family, family)
    extra = ', type.multinomial="grouped"' if family == 'multinomial_grouped' else ''
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
        elif family.startswith('multinomial'):
            rpy.r.assign('Yv', D['Y'].values.astype(float))
            rpy.r('Y = factor(as.vector(Yv))')
        else:
            rpy.r.assign('Yv', D['Y'].values.astype(float))
            rpy.r('Y = as.vector(Yv)')
        rpy.r(f'G = glmnet(X, Y, family="{R_family}", lambda=as.vector(lam), control=list(thresh=1e-14){extra})')
        if s is None:
            rpy.r('NZ = predict(G, type="nonzero")')
        else:
            rpy.r.assign('s', np.atleast_1d(s))
            rpy.r('NZ = predict(G, type="nonzero", s=as.vector(s))')
        # R's mgaussian and grouped multinomial keep the intercept as index 1
        shift = 2 if family in ['mgaussian', 'multinomial_grouped'] else 1
        def get(expr):
            n = int(rpy.r(f'length({expr})')[0])
            out = []
            for i in range(1, n + 1):
                v = np.asarray(rpy.r(f'as.integer(unlist({expr}[[{i}]]))'), int) - shift
                out.append(v[v >= 0])
            return out
        if family == 'multinomial':
            return [get(f'NZ[[{k}]]') for k in range(1, 4)]
        return get('NZ')

def check_equal(ours, theirs, family, start=0):
    if family == 'multinomial':
        assert len(ours) == len(theirs)
        for o, t in zip(ours, theirs):
            check_equal(o, t, 'gaussian', start=start)
        return
    assert len(ours) == len(theirs)
    for o, t in list(zip(ours, theirs))[start:]:
        np.testing.assert_array_equal(o, t)

FAMILIES = ['gaussian', 'irls', 'binomial', 'poisson', 'multinomial',
            'multinomial_grouped', 'mgaussian', 'cox']

@pytest.mark.parametrize('family', FAMILIES)
def test_nonzero_matches_R(Rinfo, family):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(family)
    L = get_estimator(family)
    L.nlambda = 20
    L.fit(X, D)
    lambda_values = L.lambda_values_

    # at lambda_max either fit may leave a coefficient of order 1e-16,
    # so we skip the first lambda
    check_equal(L.nonzero(),
                R_nonzero(Rinfo, family, X, D, lambda_values),
                family,
                start=1)

    # interpolated, between grid points
    s = np.sqrt(lambda_values[[3, 7]] * lambda_values[[4, 8]])
    check_equal(L.nonzero(interpolation_grid=s),
                R_nonzero(Rinfo, family, X, D, lambda_values, s=s),
                family)

@pytest.mark.parametrize('family', FAMILIES)
def test_nonzero_shapes(family):

    X, D = get_data(family)
    L = get_estimator(family)
    L.nlambda = 10
    L.fit(X, D)
    nz = L.nonzero()
    lam = L.lambda_values_[4]
    single = L.nonzero(interpolation_grid=lam)
    if family == 'multinomial':
        assert len(nz) == 3
        assert all(len(v) == L.lambda_values_.shape[0] for v in nz)
        for k in range(3):
            np.testing.assert_array_equal(single[k], nz[k][4])
            np.testing.assert_array_equal(nz[k][4], np.nonzero(L.coefs_[4][:, k])[0])
    else:
        assert len(nz) == L.lambda_values_.shape[0]
        np.testing.assert_array_equal(single, nz[4])
        coefs = L.coefs_[4]
        if coefs.ndim == 2:
            coefs = np.abs(coefs).sum(1)
        np.testing.assert_array_equal(nz[4], np.nonzero(coefs)[0])
    # the first lambda has no variables, as lambda_max is the smallest such
    if family != 'irls':
        first = nz[0] if family != 'multinomial' else nz[0][0]
        assert first.shape == (0,)
