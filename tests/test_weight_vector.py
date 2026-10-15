"""
The family is given one weight vector, the normalized weights: Cox's
deviance (coxdev) sorts the data again for each new one.
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from coxdev import CoxDeviance

from glmnet import GLMNet
from glmnet.cox import CoxFamily, CoxNetIRLS

rng = np.random.default_rng(0)
N, P = 200, 4


def _cox_data(unequal):
    X = rng.standard_normal((N, P))
    T = rng.exponential(np.exp(-X[:, 0]))
    C = rng.exponential(1.5, N)
    D = pd.DataFrame({'event': np.round(np.minimum(T, C), 1) + 0.1,
                      'status': (T <= C).astype(int),
                      'w': rng.uniform(0.5, 2, N) if unequal else np.ones(N)})
    return X, D


@pytest.mark.parametrize('unequal', [False, True])
@pytest.mark.parametrize('lambda_values', [None, np.array([0.])])
def test_cox_builds_once(monkeypatch, unequal, lambda_values):
    X, D = _cox_data(unequal)
    builds = Counter()
    reconstruct = CoxDeviance._reconstruct_coxc

    def counting(self, sample_weight):
        builds[id(self)] += 1
        return reconstruct(self, sample_weight)

    monkeypatch.setattr(CoxDeviance, '_reconstruct_coxc', counting)
    fit = CoxNetIRLS(family=CoxFamily(tie_breaking='efron'), weight_id='w',
                     lambda_values=lambda_values).fit(X, D)
    assert builds and set(builds.values()) == {1}

    # the null deviance, on the scale of the weights
    w = D['w'].values
    null = CoxDeviance(event=D['event'], status=D['status'], sample_weight=w,
                       tie_breaking='efron')(np.zeros(N), w).deviance
    assert np.allclose(fit.null_deviance_, null, rtol=1e-12)


def test_glm_deviance_scale():
    """deviance_ and null_deviance_ are on the scale of the weights, not the normalized ones."""
    X = rng.standard_normal((N, P))
    y = rng.poisson(np.exp(0.3 * X[:, 0]))
    w = rng.uniform(0.5, 2, N)
    D = pd.DataFrame({'y': y, 'w': w})
    fit = GLMNet(family=sm.families.Poisson(), response_id='y', weight_id='w',
                 lambda_values=np.array([0.])).fit(X, D)
    est = fit.reg_glm_est_
    family = sm.families.Poisson()
    mu = np.exp(fit.intercepts_[0] + X @ fit.coefs_[0])
    assert np.allclose(est.deviance_, family.deviance(y, mu, freq_weights=w), rtol=1e-10)
    ybar = (w * y).sum() / w.sum()
    assert np.allclose(fit.null_deviance_, family.deviance(y, np.full(N, ybar), freq_weights=w),
                       rtol=1e-10)
