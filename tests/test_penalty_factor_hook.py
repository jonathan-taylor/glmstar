from dataclasses import dataclass

import numpy as np
import pytest
import statsmodels.api as sm

from glmnet import GLMNet, GaussNet, LogNet

rng = np.random.default_rng(0)
N, P = 100, 8


def _data(binary=False):
    X = rng.standard_normal((N, P))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.standard_normal(N)
    return X, y


FIXED_PF = np.r_[0.5, 2., 1., 1., 0., 1., 3., 1.]


@dataclass
class FixedPFGaussNet(GaussNet):
    def get_penalty_factor(self, X, y):
        return FIXED_PF.copy()


@dataclass
class FixedPFLogNet(LogNet):
    def get_penalty_factor(self, X, y):
        return FIXED_PF.copy()


@dataclass
class FixedPFGLMNet(GLMNet):
    def get_penalty_factor(self, X, y):
        return np.r_[0.5, 2., 1., 1., 0.5, 1., 3., 1.]


@dataclass
class AdaptiveGaussNet(GaussNet):
    # adaptive lasso: penalty factors 1 / |least squares coefficients|
    def get_penalty_factor(self, X, y):
        X1 = np.column_stack([np.ones(X.shape[0]), X])
        beta = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
        return 1 / np.fabs(beta)


@pytest.mark.parametrize('cls,base,binary', [(FixedPFGaussNet, GaussNet, False),
                                            (FixedPFLogNet, LogNet, True)])
def test_hook_matches_fixed(cls, base, binary):
    X, y = _data(binary)
    hooked = cls().fit(X, y)
    fixed = base(penalty_factor=FIXED_PF.copy()).fit(X, y)
    np.testing.assert_allclose(hooked.lambda_values_, fixed.lambda_values_)
    np.testing.assert_allclose(hooked.coefs_, fixed.coefs_)
    np.testing.assert_allclose(hooked.intercepts_, fixed.intercepts_)
    # the constructor parameter is untouched
    assert hooked.penalty_factor is None
    np.testing.assert_allclose(hooked.penalty_factor_, FIXED_PF)


def test_hook_matches_fixed_irls():
    X, y = _data(True)
    pf = np.r_[0.5, 2., 1., 1., 0.5, 1., 3., 1.]
    family = sm.families.Binomial()
    hooked = FixedPFGLMNet(family=family, nlambda=20).fit(X, y)
    fixed = GLMNet(family=family, nlambda=20, penalty_factor=pf).fit(X, y)
    np.testing.assert_allclose(hooked.lambda_values_, fixed.lambda_values_)
    np.testing.assert_allclose(hooked.coefs_, fixed.coefs_)
    assert hooked.penalty_factor is None


def test_hook_infinite_excludes():
    X, y = _data()

    @dataclass
    class InfGaussNet(GaussNet):
        def get_penalty_factor(self, X, y):
            pf = np.ones(X.shape[1])
            pf[[2, 5]] = np.inf
            return pf

    fit = InfGaussNet().fit(X, y)
    assert np.all(fit.coefs_[:, [2, 5]] == 0)
    np.testing.assert_array_equal(np.sort(fit.excluded_), [2, 5])
    fixed = GaussNet(exclude=[2, 5]).fit(X, y)
    np.testing.assert_allclose(fit.coefs_, fixed.coefs_)


def test_hook_rerun_on_folds():
    X, y = _data()
    calls = []

    @dataclass
    class LoggedGaussNet(AdaptiveGaussNet):
        def get_penalty_factor(self, X, y):
            calls.append(X.shape[0])
            return super().get_penalty_factor(X, y)

    fit = LoggedGaussNet().fit(X, y)
    assert calls == [N]
    calls.clear()
    fit.cross_validation_path(X, y, cv=5)
    assert calls == [N * 4 // 5] * 5


def test_hook_matches_R(Rinfo):
    # R glmnet >= 5.1 accepts a function for penalty.factor
    X, y = _data()
    rpy = Rinfo['rpy']
    with rpy.conversion.localconverter(Rinfo['np_cv_rules']):
        rpy.r.assign('x', X)
        rpy.r.assign('y', y)
        rpy.r('''
        pf <- function(x, y, ...) 1 / abs(coef(lm(y ~ x))[-1])
        fit <- glmnet(x, y, penalty.factor = pf)
        lambda <- fit$lambda
        coefR <- t(as.matrix(coef(fit)))
        ''')
        lambdaR = np.asarray(rpy.r['lambda'])
        coefR = np.asarray(rpy.r['coefR'])
    fit = AdaptiveGaussNet(lambda_values=lambdaR).fit(X, y)
    np.testing.assert_allclose(fit.coefs_, coefR[:, 1:], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(fit.intercepts_, coefR[:, 0], rtol=1e-4, atol=1e-6)
