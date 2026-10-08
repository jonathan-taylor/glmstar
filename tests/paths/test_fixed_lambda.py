import numpy as np
import pandas as pd
import pytest

from glmnet.paths import (GaussNet, LogNet, FishNet, CoxNet,
                          MultiGaussNet, MultiClassNet)
from glmnet.cox import CoxFamily, CoxState
from glmnet.glm import GLMState

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

@pytest.mark.parametrize('cls', [MultiGaussNet, MultiClassNet])
def test_get_fixed_lambda_multi(cls, n=100, p=5):

    X = rng.standard_normal((n, p))
    if cls is MultiGaussNet:
        Y = X[:, :2] + rng.standard_normal((n, 2))
    else:
        Y = rng.choice(3, n)
    L = cls().fit(X, Y)
    with pytest.raises(NotImplementedError, match='single lambda'):
        L.get_fixed_lambda(L.lambda_values_[5])
