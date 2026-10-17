"""
The lambda scale of the gaussian family for alpha < 1.

R's glmnet has two gaussian fits: ``family="gaussian"`` fits the response
divided by its weighted standard deviation `s_y` at `lambda / s_y`, which
divides the ridge part of the penalty by `s_y`; ``family=gaussian()`` (the
IRLS path, ``glmnet.path``) does not rescale. GaussNet is the former, the
IRLS GLMNet and RegGLM the latter. For the lasso (alpha=1) they agree.
GaussNet's `get_fixed_lambda` gives the RegGLM solving GaussNet's problem.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import pytest

from sklearn.linear_model import ElasticNet

from glmnet import GaussNet, GLMNet
from glmnet.regularized_glm import RegGLM

rng = np.random.default_rng(3)
n, p = 100, 5
X = rng.standard_normal((n, p)) * [1, 2, 3, 1, 1]
y = 3 + X @ [1, -1, 0.5, 0, 0] + 2 * rng.standard_normal(n)
W = rng.uniform(0.5, 2, n)
O = 0.3 * rng.standard_normal(n)
LAM = 0.3


def _data(use_weights, use_offset):
    D = pd.DataFrame({'y': y,
                      'w': W if use_weights else np.ones(n),
                      'o': O if use_offset else np.zeros(n)})
    args = dict(response_id='y', weight_id='w', offset_id='o')
    return D, args


def _R_coef(Rinfo, D, family, alpha, standardize):
    rpy = Rinfo['rpy']
    FloatVector = Rinfo['FloatVector']
    rpy.globalenv['x'] = rpy.r.matrix(FloatVector(X.T.flatten()), nrow=n, ncol=p)
    for name in ['y', 'w', 'o']:
        rpy.globalenv[name] = FloatVector(np.asarray(D[name], float))
    rpy.r(f'f <- glmnet(x, y, weights=w, offset=o, alpha={alpha}, lambda={LAM}, '
          f'standardize={"TRUE" if standardize else "FALSE"}, family={family}, '
          f'thresh=1e-14)')
    return np.asarray(rpy.r('as.numeric(coef(f))'))


@pytest.mark.parametrize('mixing', [0, 0.5, 1])
def test_gaussian_lambda_scale_R(Rinfo, mixing, standardize, use_weights, use_offset):
    D, args = _data(use_weights, use_offset)
    kw = dict(alpha=mixing, standardize=standardize, **args)

    path = GaussNet(lambda_values=[LAM], **kw).fit(X, D)
    irls = GLMNet(family=sm.families.Gaussian(), lambda_values=[LAM], **kw).fit(X, D)
    reg = RegGLM(lambda_val=LAM, **kw).fit(X, D)

    R_string = _R_coef(Rinfo, D, '"gaussian"', mixing, standardize)
    R_family = _R_coef(Rinfo, D, 'gaussian()', mixing, standardize)

    np.testing.assert_allclose(np.r_[path.intercepts_[0], path.coefs_[0]], R_string, atol=1e-5)
    np.testing.assert_allclose(np.r_[irls.intercepts_[0], irls.coefs_[0]], R_family, atol=1e-5)
    np.testing.assert_allclose(np.r_[reg.intercept_, reg.coef_], R_family, atol=1e-5)
    if mixing < 1:
        assert np.max(np.abs(R_string - R_family)) > 1e-3


# standardize, fit_intercept, use_weights and use_offset are parametrized in conftest.py
@pytest.mark.parametrize('mixing', [0, 0.5, 1])
def test_gaussnet_get_fixed_lambda(mixing, standardize, fit_intercept, use_weights, use_offset):
    D, args = _data(use_weights, use_offset)
    path = GaussNet(alpha=mixing, standardize=standardize,
                    fit_intercept=fit_intercept, **args).fit(X, D)
    for k in [5, 15]:
        estimator, _ = path.get_fixed_lambda(path.lambda_values_[k])
        estimator.fit(X, D)
        np.testing.assert_allclose(estimator.coef_, path.coefs_[k], atol=1e-4)
        np.testing.assert_allclose(estimator.intercept_, path.intercepts_[k], atol=1e-4)


def test_ridge_sklearn():
    # the ridge fits are sklearn's ElasticNet at lambda / s_y (GaussNet) and lambda (RegGLM)
    s_y = y.std()
    path = GaussNet(alpha=0, lambda_values=[LAM], standardize=False).fit(X, y)
    reg = RegGLM(alpha=0, lambda_val=LAM, standardize=False).fit(X, y)
    for lam, coef in [(LAM / s_y, path.coefs_[0]), (LAM, reg.coef_)]:
        sk = ElasticNet(alpha=lam, l1_ratio=0, tol=1e-12, max_iter=100000).fit(X, y)
        np.testing.assert_allclose(coef, sk.coef_, atol=1e-5)
