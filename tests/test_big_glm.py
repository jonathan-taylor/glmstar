import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet import (GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet,
                    CoxNet, GLMNet, big_glm)
from glmnet.glmnet import GLMNetControl
from glmnet.paths.fastnet import FastNetControl

N, P = 150, 4

FAMILIES = {'gaussian': GaussNet,
            'binomial': LogNet,
            'poisson': FishNet,
            'multinomial': MultiClassNet,
            'mgaussian': MultiGaussNet,
            'cox': CoxNet}


def _data(family, use_wts, use_off):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, P))
    eta = X @ np.array([0.6, -0.4, 0.3, 0.])
    df = pd.DataFrame({'w': rng.uniform(0.5, 2, N), 'o': rng.normal(0, 0.2, N)})
    if family == 'gaussian':
        df['y'] = eta + rng.standard_normal(N)
    elif family == 'binomial':
        df['y'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    elif family == 'poisson':
        df['y'] = rng.poisson(np.exp(eta / 2))
    elif family == 'multinomial':
        logits = np.column_stack([eta, -eta, np.zeros(N)])
        prob = np.exp(logits) / np.exp(logits).sum(1)[:, None]
        df['y'] = [rng.choice(3, p=p_) for p_ in prob]
    elif family == 'mgaussian':
        df['y1'] = eta + rng.standard_normal(N)
        df['y2'] = -eta + rng.standard_normal(N)
    else:
        T = rng.exponential(np.exp(-eta))
        C = rng.exponential(1.5, size=N)
        df['event'] = np.minimum(T, C)
        df['status'] = (T <= C).astype(int)
    args = {}
    if family not in ['cox', 'mgaussian']:
        args['response_id'] = 'y'
    if family == 'mgaussian':
        args['response_id'] = ['y1', 'y2']
    if use_wts:
        args['weight_id'] = 'w'
    if use_off:
        args['offset_id'] = 'o'
    return X, df, args


def _R_big_glm(Rinfo, X, df, family, use_wts, use_off, extra=''):
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('x', X)
        rpy.r.assign('df', df)
    if family == 'cox':
        rpy.r('y <- survival::Surv(df$event, df$status)')
    elif family == 'mgaussian':
        rpy.r('y <- as.matrix(df[, c("y1", "y2")])')
    else:
        rpy.r('y <- df$y')
    args = f'family = "{family}", thresh = 1e-14'
    if use_wts:
        args += ', weights = df$w'
    if use_off:
        args += ', offset = df$o'
    rpy.r(f'fit <- glmnet::bigGlm(x, y, {args}{extra})')
    with Rinfo['np_cv_rules'].context():
        if family in ['multinomial', 'mgaussian']:
            beta = np.asarray(rpy.r('sapply(fit$beta, function(b) as.numeric(as.matrix(b)))'))
            a0 = np.asarray(rpy.r('as.numeric(fit$a0)'))
        else:
            beta = np.asarray(rpy.r('as.numeric(as.matrix(fit$beta))'))
            a0 = np.asarray(rpy.r('as.numeric(fit$a0)'))
    return beta, a0


def _control(family):
    return FastNetControl(thresh=1e-14)


@pytest.mark.parametrize('family', list(FAMILIES))
@pytest.mark.parametrize('use_wts', [False, True])
@pytest.mark.parametrize('use_off', [False, True])
def test_big_glm_matches_R(Rinfo, family, use_wts, use_off):
    if use_off and family in ['multinomial', 'mgaussian']:
        pytest.skip('offsets for multiple responses are a matrix')
    X, df, args = _data(family, use_wts, use_off)
    est = FAMILIES[family](control=_control(family), **args)
    fit = big_glm(est, X, df)
    beta_R, a0_R = _R_big_glm(Rinfo, X, df, family, use_wts, use_off)

    assert fit.coefs_.shape[0] == 1
    np.testing.assert_allclose(fit.lambda_values_, [0.])
    np.testing.assert_allclose(fit.coefs_[0], beta_R, rtol=1e-5, atol=1e-6)
    if family != 'cox':
        np.testing.assert_allclose(np.squeeze(fit.intercepts_[0]), a0_R, rtol=1e-5, atol=1e-6)
    nonzero = fit.coefs_[0] != 0
    if nonzero.ndim == 2:
        nonzero = nonzero.any(1)
    assert fit.summary_['Degrees of Freedom'].iloc[0] == nonzero.sum()
    # the estimator passed in is not modified
    assert est.lambda_values is None


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_big_glm_exclude_limits_matches_R(Rinfo, family):
    X, df, args = _data(family, True, False)
    upper = np.array([0.1, np.inf, np.inf, np.inf])
    est = FAMILIES[family](control=_control(family), exclude=[3], upper_limits=upper, **args)
    fit = big_glm(est, X, df)
    # bigGlm appends a column to x, so R needs vectors of length nvars + 1;
    # it also replaces any `exclude` argument by that column, so exclude
    # variable 4 by an infinite penalty factor instead
    beta_R, a0_R = _R_big_glm(Rinfo, X, df, family, True, False,
                              extra=(', penalty.factor = c(1, 1, 1, Inf, 1)'
                                     ', upper.limits = c(0.1, Inf, Inf, Inf, Inf)'))
    np.testing.assert_allclose(fit.coefs_[0], beta_R, rtol=1e-5, atol=1e-6)
    assert fit.coefs_[0, 3] == 0
    assert fit.coefs_[0, 0] <= 0.1 + 1e-10


@pytest.mark.parametrize('family', list(FAMILIES))
def test_big_glm_path(family):
    X, df, args = _data(family, True, False)
    est = FAMILIES[family](control=_control(family), **args)
    direct = big_glm(est, X, df)
    via_path = big_glm(est, X, df, path=True)
    assert via_path.coefs_.shape == direct.coefs_.shape
    np.testing.assert_allclose(via_path.lambda_values_, [0.])
    np.testing.assert_allclose(via_path.coefs_, direct.coefs_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(via_path.lambda_values, [0.])
    assert via_path.summary_.shape[0] == 1


@pytest.mark.parametrize('use_path', [False, True])
def test_big_glm_irls(use_path):
    # GLMNet's IRLS path stops early on its deviance criterion, before lambda = 0
    X, df, args = _data('binomial', True, True)
    irls = big_glm(GLMNet(family=sm.families.Binomial(),
                          control=GLMNetControl(thresh=1e-14), **args), X, df, path=use_path)
    fast = big_glm(LogNet(control=FastNetControl(thresh=1e-14), **args), X, df)
    np.testing.assert_allclose(irls.lambda_values_, [0.])
    np.testing.assert_allclose(irls.coefs_, fast.coefs_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(irls.intercepts_, fast.intercepts_, rtol=1e-5, atol=1e-6)


def test_big_glm_class_and_predict():
    X, df, args = _data('gaussian', False, False)
    fit = big_glm(GaussNet, X, df['y'].values)
    ls = np.linalg.lstsq(np.column_stack([np.ones(N), X]), df['y'].values, rcond=None)[0]
    np.testing.assert_allclose(fit.coefs_[0], ls[1:], rtol=1e-5)
    np.testing.assert_allclose(np.squeeze(fit.predict(X)), np.column_stack([np.ones(N), X]) @ ls,
                               rtol=1e-5, atol=1e-6)
