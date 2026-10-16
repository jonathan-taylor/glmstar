"""
Refitting an estimator to other data gives the fit of a fresh one, as
scikit-learn expects of `fit` (e.g. ISLP's FeatureSelector refits one
estimator for each candidate model). The design and the family (Cox's holds
the event data) used to be kept from the first fit.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from glmnet import GLM, GLMNet, GaussNet, LogNet, FishNet, CoxNetIRLS
from glmnet.glm import GaussianGLM, BinomialGLM
from glmnet.regularized_glm import RegGLM
from glmnet.cox import CoxLM, CoxFamilySpec
from glmnet.elnet import ElNet
from glmnet.paths.coxnet import CoxNet

rng = np.random.default_rng(0)
n, p = 200, 5
X1 = rng.standard_normal((n, p))
X2 = 3 * rng.standard_normal((n, p)) + 1
y1 = X1[:, 0] + rng.standard_normal(n)
y2 = X2[:, 1] + rng.standard_normal(n)
count1 = rng.poisson(1, n)
count2 = rng.poisson(np.exp(X2[:, 0] / 3))


def _surv(X):
    t = rng.exponential(np.exp(-X[:, 0] / 3))
    c = rng.exponential(3, n)
    return pd.DataFrame({'event': np.minimum(t, c), 'status': (t <= c).astype(float)})


Y1, Y2 = _surv(X1), _surv(X2)
LAM = dict(lambda_values=np.array([0.5, 0.1, 0.01]))

CASES = {
    'GLM': (lambda: GLM(), (X1, y1), (X2, y2)),
    'GLM, other columns': (lambda: GLM(), (X1[:, :2], y1), (X2, y2)),
    'GaussianGLM': (lambda: GaussianGLM(), (X1, y1), (X2, y2)),
    'BinomialGLM': (lambda: BinomialGLM(), (X1, y1 > 0), (X2, y2 > 0)),
    'CoxLM': (lambda: CoxLM(family=CoxFamilySpec(event_data=Y1)), (X1, Y1), (X2, Y2)),
    'RegGLM': (lambda: RegGLM(lambda_val=0.05), (X1, y1), (X2, y2)),
    'ElNet': (lambda: ElNet(lambda_val=0.05), (X1, y1), (X2, y2)),
    'GLMNet': (lambda: GLMNet(**LAM), (X1, y1), (X2, y2)),
    'GLMNet, own path': (lambda: GLMNet(nlambda=10), (X1, y1), (X2, y2)),
    'GaussNet': (lambda: GaussNet(**LAM), (X1, y1), (X2, y2)),
    'LogNet': (lambda: LogNet(**LAM), (X1, y1 > 0), (X2, y2 > 0)),
    'FishNet': (lambda: FishNet(**LAM), (X1, count1), (X2, count2)),
    'CoxNet': (lambda: CoxNet(**LAM), (X1, Y1), (X2, Y2)),
    'CoxNetIRLS': (lambda: CoxNetIRLS(family=CoxFamilySpec(event_data=Y1), nlambda=10),
                   (X1, Y1), (X2, Y2)),
}


def _coef(model):
    return np.asarray(model.coefs_ if hasattr(model, 'coefs_') else model.coef_)


@pytest.mark.parametrize('name', list(CASES))
def test_refit(name):
    make, first, second = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        refit = make()
        refit.fit(*first)
        refit.fit(*second)
        fresh = make().fit(*second)
    np.testing.assert_allclose(_coef(refit), _coef(fresh), rtol=1e-10, atol=1e-12)
    for attr in ['intercept_', 'intercepts_', 'lambda_values_', 'null_deviance_', 'deviance_']:
        if hasattr(fresh, attr):
            np.testing.assert_allclose(getattr(refit, attr), getattr(fresh, attr),
                                       rtol=1e-10, atol=1e-12)
