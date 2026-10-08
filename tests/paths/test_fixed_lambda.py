import numpy as np
import pandas as pd
import pytest

from glmnet.paths import (GaussNet, LogNet, FishNet, CoxNet,
                          MultiGaussNet, MultiClassNet)
from glmnet.cox import CoxFamily, CoxState
from glmnet.glm import GLMState
from glmnet.paths.fastnet import FixedLambdaMultiNet, MultiState

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
    args = {}
    if family != 'cox':
        args['response_id'] = 'Y'
    if use_weights:
        D['weight'] = rng.uniform(0.5, 2, n)
        args['weight_id'] = 'weight'
    if use_offset:
        D['offset'] = 0.3 * rng.standard_normal(n)
        args['offset_id'] = 'offset'
    return X, D, args

ESTIMATORS = {'gaussian': GaussNet,
              'binomial': LogNet,
              'poisson': FishNet}

# the single lambda fits iterate to their own convergence criteria (IRLS for
# the GLMs), so agree with the C++ path only to within those tolerances
TOL = {'gaussian': 1e-5, 'binomial': 1e-4, 'poisson': 1e-4, 'cox': 2e-3}

@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_get_fixed_lambda(family, use_weights, use_offset, n=150, p=8):

    X, D, args = get_data(family, n, p, use_weights, use_offset)
    if family == 'cox':
        L = CoxNet(family=CoxFamily(tie_breaking='breslow', event_id='stop',
                                    status_id='status'), **args)
    else:
        L = ESTIMATORS[family](**args)
    L.fit(X, D)

    state_class = CoxState if family == 'cox' else GLMState
    assert isinstance(L.state_, state_class)
    assert np.allclose(L.state_.coef, L.coefs_[-1])

    for k in [5, 15]:
        estimator, state = L.get_fixed_lambda(L.lambda_values_[k])
        assert isinstance(state, state_class)
        # the warm start is the path solution at a path value
        assert np.allclose(state.coef, L.coefs_[k])
        assert np.allclose(state.intercept, L.intercepts_[k])

        estimator.fit(X, D, warm_state=state)
        assert np.allclose(estimator.coef_, L.coefs_[k], rtol=0, atol=TOL[family])
        assert np.allclose(estimator.intercept_, L.intercepts_[k], rtol=0, atol=TOL[family])

# ---- multiple responses ----

MULTI = {'mgaussian': (MultiGaussNet, {}),
         'multinomial': (MultiClassNet, {}),
         'grouped': (MultiClassNet, {'grouped': True})}

def get_multi_data(family, n, p, q, use_weights, use_offset):
    X = rng.standard_normal((n, p))
    B = np.zeros((p, q))
    B[:3] = rng.standard_normal((3, q))
    eta = X @ B
    if family == 'mgaussian':
        Y = eta + rng.standard_normal((n, q))
        response_id = [f'Y{j}' for j in range(q)]
        D = pd.DataFrame(Y, columns=response_id)
    else:
        prob = np.exp(eta) / np.exp(eta).sum(1)[:, None]
        Y = np.array([rng.choice(q, p=p_) for p_ in prob])
        response_id = 'Y'
        D = pd.DataFrame({'Y': Y})
    args = {'response_id': response_id}
    if use_weights:
        D['weight'] = rng.uniform(0.5, 2, n)
        args['weight_id'] = 'weight'
    if use_offset:
        offset_id = [f'offset{j}' for j in range(q)]
        for l in offset_id:
            D[l] = 0.3 * rng.standard_normal(n)
        args['offset_id'] = offset_id
    return X, D, args

def get_multi_fit(family, X, D, args):
    cls, kw = MULTI[family]
    return cls(**kw, **args).fit(X, D)

@pytest.mark.parametrize('family', list(MULTI))
def test_get_fixed_lambda_multi(family, use_weights, use_offset, n=150, p=6, q=3):

    X, D, args = get_multi_data(family, n, p, q, use_weights, use_offset)
    L = get_multi_fit(family, X, D, args)

    assert isinstance(L.state_, MultiState)
    assert np.allclose(L.state_.coef, L.coefs_[-1])

    for k in [5, 15]:
        estimator, state = L.get_fixed_lambda(L.lambda_values_[k])
        assert isinstance(estimator, FixedLambdaMultiNet)
        assert isinstance(state, MultiState)
        # the warm start is the path solution at a path value
        assert np.allclose(state.coef, L.coefs_[k])
        assert np.allclose(state.intercept, L.intercepts_[k])

        estimator.fit(X, D, warm_state=state)
        assert estimator.coef_.shape == (p, q)
        assert estimator.intercept_.shape == (q,)
        # the same path down to lambda_values_[k]
        assert np.allclose(estimator.coef_, L.coefs_[k])
        assert np.allclose(estimator.intercept_, L.intercepts_[k])

        assert np.allclose(estimator.predict(X),
                           L.predict(X, interpolation_grid=[L.lambda_values_[k]])[:, 0])

    # between path values: the same as refitting the path
    lam = 0.5 * (L.lambda_values_[7] + L.lambda_values_[8])
    estimator, _ = L.get_fixed_lambda(lam)
    estimator.fit(X, D)
    coefs, intercepts = L.exact_coefs(X, D, lam)
    assert np.allclose(estimator.coef_, coefs)
    assert np.allclose(estimator.intercept_, intercepts)

    # the estimator is not fitted in place and leaves L unchanged
    assert L.lambda_values is None

    with pytest.raises(ValueError, match='non-negative'):
        L.get_fixed_lambda(-1.)

def get_R_multi(Rinfo, family, X, D, args, s):
    """R's coef(glmnet(...), s=s) and coef(..., exact=TRUE), each of shape (len(s), p, q), and intercepts."""
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    with np_cv_rules.context():
        rpy.r('rm(list=intersect(c("W", "O"), ls()))')
        rpy.r.assign('X', X)
        rpy.r.assign('s', s)
        if family == 'mgaussian':
            rpy.r.assign('Y', D[args['response_id']].values)
            fam = '"mgaussian"'
        else:
            rpy.r.assign('Y', D['Y'].values.astype(float))
            rpy.r('Y = as.factor(as.vector(Y))')
            fam = '"multinomial"'
            if family == 'grouped':
                fam += ', type.multinomial="grouped"'
        extra = ''
        if 'weight_id' in args:
            rpy.r.assign('W', D['weight'].values)
            extra += ', weights=as.vector(W)'
        if 'offset_id' in args:
            rpy.r.assign('O', D[args['offset_id']].values)
            extra += ', offset=O'
        rpy.r(f'''
suppressMessages(library(glmnet))
G = glmnet(X, Y, family={fam}{extra})
interp = lapply(coef(G, s=as.vector(s)), as.matrix)
exact = lapply(coef(G, s=as.vector(s), exact=TRUE, x=X, y=Y{extra}), as.matrix)
q = length(interp)
''')
        q = int(rpy.r('q')[0])
        out = []
        for name in ['interp', 'exact']:
            # each of shape (p + 1, len(s))
            B = np.array([np.asarray(rpy.r(f'{name}[[{j + 1}]]')) for j in range(q)])
            B = np.transpose(B, [2, 1, 0])
            out.append((B[:, 1:], B[:, 0]))
    return out

@pytest.mark.parametrize('family', list(MULTI))
def test_get_fixed_lambda_multi_R(Rinfo, family, use_weights, use_offset, n=150, p=6, q=3):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, args = get_multi_data(family, n, p, q, use_weights, use_offset)
    L = get_multi_fit(family, X, D, args)
    lam = L.lambda_values_
    # on a path value and between path values
    s = np.array([lam[10], 0.5 * (lam[3] + lam[4]), 0.7 * lam[20] + 0.3 * lam[21]])

    (R_interp, R_interp_int), (R_exact, R_exact_int) = get_R_multi(Rinfo, family, X, D, args, s)

    for i, s_ in enumerate(s):
        estimator, state = L.get_fixed_lambda(s_)
        # the warm start interpolates the path, as R's coef(..., exact=FALSE)
        assert np.allclose(state.coef, R_interp[i], rtol=1e-5, atol=1e-5)
        assert np.allclose(state.intercept, R_interp_int[i], rtol=1e-5, atol=1e-5)
        # the fit is exact, as R's coef(..., exact=TRUE)
        estimator.fit(X, D)
        assert np.allclose(estimator.coef_, R_exact[i], rtol=1e-5, atol=1e-5)
        assert np.allclose(estimator.intercept_, R_exact_int[i], rtol=1e-5, atol=1e-5)
