import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import glmnet.irls as irls
from glmnet import GLMNet
from glmnet.cox import CoxFamily, CoxNetIRLS, CoxState
from glmnet.glm import GLMState
from glmnet.glmnet import GLMNetControl

rng = np.random.default_rng(0)
N, P = 300, 5
TIGHT = GLMNetControl(epsnr=1e-12, thresh=1e-14)


def _data(family):
    # features on different scales and off center, so that the standardized
    # coordinates of the design differ from the original ones
    X = rng.standard_normal((N, P)) * rng.uniform(0.2, 5, P) + rng.uniform(-2, 2, P)
    eta = 0.3 + 0.5 * X[:, 0] / X[:, 0].std() - 0.4 * X[:, 1] / X[:, 1].std()
    if family == 'binomial':
        return X, pd.DataFrame({'y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    if family == 'poisson':
        return X, pd.DataFrame({'y': rng.poisson(np.exp(eta))})
    if family == 'cox':
        T = rng.exponential(np.exp(-eta))
        C = rng.exponential(1.5, N)
        return X, pd.DataFrame({'event': np.round(np.minimum(T, C), 1) + 0.1,
                                'status': (T <= C).astype(int)})
    return X, pd.DataFrame({'y': eta + rng.standard_normal(N)})


def _estimator(family, **kwargs):
    if family == 'cox':
        return CoxNetIRLS(family=CoxFamily(tie_breaking='efron'), control=TIGHT, **kwargs)
    fam = {'gaussian': sm.families.Gaussian(),
           'binomial': sm.families.Binomial(),
           'poisson': sm.families.Poisson()}[family]
    return GLMNet(family=fam, response_id='y', control=TIGHT, **kwargs)


def _state(family, coef, intercept):
    return (CoxState if family == 'cox' else GLMState)(coef=coef, intercept=intercept)


@pytest.fixture
def steps(monkeypatch):
    """
    The coefficients each quasi-Newton step of IRLS starts from, on the original
    scale, other than in GLMNet's initial fit at lambda = big.
    """
    starts = []
    step = irls.quasi_newton_step

    def recording(regularizer, family, design, y, offset, weights, state, *args):
        if regularizer.lambda_val < 1e30:
            starts.append(state.coef / design.scaling_)
        return step(regularizer, family, design, y, offset, weights, state, *args)

    monkeypatch.setattr(irls, 'quasi_newton_step', recording)
    return starts


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_warm_start_at_solution(family, steps):
    """A warm start at the solution, on the original scale: IRLS starts there and takes one step."""
    X, D = _data(family)
    cold = _estimator(family, lambda_values=np.array([0.])).fit(X, D)
    coef, intercept = cold.coefs_[0], cold.intercepts_[0]

    steps.clear()
    warm = _estimator(family, lambda_values=np.array([0.])).fit(
        X, D, warm_state=_state(family, coef, intercept))
    assert np.allclose(steps[0], coef, rtol=0, atol=1e-12)
    assert len(steps) == 1
    # the cold fit stops a little short of the solution, by the relative change
    # in the objective
    assert np.allclose(warm.coefs_[0], coef, rtol=0, atol=1e-6)
    assert np.allclose(warm.intercepts_[0], intercept, rtol=0, atol=1e-6)


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_get_fixed_lambda_warm_start(family, steps):
    """The state from get_fixed_lambda at a path value starts its estimator at the solution."""
    X, D = _data(family)
    path = _estimator(family, nlambda=20).fit(X, D)
    k = 10
    estimator, state = path.get_fixed_lambda(path.lambda_values_[k])

    steps.clear()
    estimator.fit(X, D, warm_state=state)
    assert np.allclose(steps[0], path.coefs_[k], rtol=0, atol=1e-12)
    # Cox's IRLS converges linearly from there, so it may take more than one step
    assert np.allclose(estimator.coef_, path.coefs_[k], rtol=0, atol=1e-4)
    assert np.allclose(estimator.intercept_, path.intercepts_[k], rtol=0, atol=1e-4)
