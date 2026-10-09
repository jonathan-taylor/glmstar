import numpy as np
import pandas as pd
import pytest

from glmnet.paths import CoxNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.cox import CoxFamily, CoxNetIRLS
from glmnet.glmnet import GLMNetControl

N, P = 120, 6


def get_data(seed=0):
    # rounded times, so there are tied events
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    eta = X[:, 0] - 0.5 * X[:, 1]
    T = rng.exponential(np.exp(-eta))
    C = rng.exponential(1.5, size=N)
    D = pd.DataFrame({'stop': np.round(np.minimum(T, C), 1) + 0.1,
                      'status': (T <= C).astype(int),
                      'offset': 0.3 * rng.standard_normal(N)})
    return X, D


def coxnet(**args):
    return CoxNet(family=CoxFamily(event_id='stop', status_id='status', **args),
                  control=FastNetControl(thresh=1e-14))


def coxnet_irls(**args):
    return CoxNetIRLS(family=CoxFamily(event_id='stop', status_id='status', **args),
                      nlambda=20, control=GLMNetControl(thresh=1e-12))


@pytest.mark.parametrize('estimator', [coxnet, coxnet_irls])
def test_cox_prediction_types(estimator):
    X, D = get_data()
    L = estimator().fit(X, D.drop(columns='offset'))
    link = L.predict(X, prediction_type='link')
    np.testing.assert_allclose(L.predict(X), link)
    np.testing.assert_allclose(L.predict(X, prediction_type='response'), np.exp(link))
    # the offset is added before exponentiating
    o = D['offset'].values
    np.testing.assert_allclose(L.predict(X, prediction_type='response', offset=o),
                               np.exp(link + o[:, None]))
    with pytest.raises(ValueError, match="'link' or 'response'"):
        L.predict(X, prediction_type='class')


def test_default_ties_is_breslow():
    assert CoxFamily().tie_breaking == 'breslow'
    X, D = get_data()
    L = coxnet().fit(X, D)
    assert L.family.tie_breaking == 'breslow'


def assign_R(Rinfo, X, D):
    rpy = Rinfo['rpy']
    rpy.r.assign('X', X)
    rpy.r.assign('st', D['stop'].values)
    rpy.r.assign('d', D['status'].values.astype(float))
    rpy.r('suppressMessages({library(glmnet); library(survival)}); '
          'Y = Surv(as.vector(st), as.vector(d))')


def test_cox_defaults_match_R(Rinfo):
    # with tied events: R's default cox.ties="breslow", and predict's
    # type="link" (default) and type="response" (relative risk)
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    X, D = get_data()
    L = coxnet().fit(X, D)
    with Rinfo['np_cv_rules'].context():
        assign_R(Rinfo, X, D)
        rpy.r.assign('lam', L.lambda_values_)
        rpy.r('fit = glmnet(X, Y, family="cox", lambda=lam, thresh=1e-14)')
        coef_R = np.asarray(rpy.r('as.matrix(fit$beta)')).T
        link_R = np.asarray(rpy.r('predict(fit, X)'))
        response_R = np.asarray(rpy.r('predict(fit, X, type="response")'))
        rpy.r('fit_efron = glmnet(X, Y, family="cox", lambda=lam, thresh=1e-14, cox.ties="efron")')
        coef_efron_R = np.asarray(rpy.r('as.matrix(fit_efron$beta)')).T
    nlam = coef_R.shape[0]
    # (skip lambda_max, where R solves exactly at the boundary)
    np.testing.assert_allclose(L.coefs_[1:nlam], coef_R[1:], rtol=1e-5, atol=1e-6)
    # the data have ties, so efron differs
    assert np.abs(coef_efron_R - coef_R).max() > 1e-3
    np.testing.assert_allclose(L.predict(X)[:, 1:nlam], link_R[:, 1:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(L.predict(X, prediction_type='response')[:, 1:nlam], response_R[:, 1:],
                               rtol=1e-5, atol=1e-6)


def test_survfit_default_is_efron(Rinfo):
    # R's survfit.coxnet always uses Efron's hazard estimate, here for a
    # model fit with the default (breslow) ties
    X, D = get_data()
    L = coxnet().fit(X, D)
    lam = L.lambda_values_[10]
    curves = L.survfit(X, D, lambda_val=lam)
    efron = L.survfit(X, D, lambda_val=lam, tie_breaking='efron')
    np.testing.assert_allclose(curves.cumhaz, efron.cumhaz)
    breslow = L.survfit(X, D, lambda_val=lam, tie_breaking='breslow')
    assert np.abs(breslow.cumhaz - curves.cumhaz).max() > 1e-6

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        assign_R(Rinfo, X, D)
        rpy.r.assign('s', lam)
        rpy.r('G = glmnet(X, Y, family="cox", thresh=1e-14); S = survfit(G, s=s, x=X, y=Y)')
        cumhaz_R = np.asarray(rpy.r('as.numeric(S$cumhaz)'))
        time_R = np.asarray(rpy.r('as.numeric(S$time)'))
    np.testing.assert_allclose(curves.time, time_R)
    np.testing.assert_allclose(np.asarray(curves.cumhaz).reshape(-1), cumhaz_R, rtol=1e-5, atol=1e-8)


def test_cv_uses_linear_predictor():
    # cross_validation_path scores the linear predictor (plus the held-out offset)
    X, D = get_data()

    def est():
        return CoxNet(family=CoxFamily(event_id='stop', status_id='status'), offset_id='offset',
                      control=FastNetControl(thresh=1e-14))

    L = est().fit(X, D)
    train, test = np.arange(80), np.arange(80, N)
    predictions, _ = L.cross_validation_path(X, D, cv=[(train, test), (test, train)])
    L_tr = est().fit(X[train], D.iloc[train])
    expected = L_tr.predict(X[test], prediction_type='link', offset=D['offset'].values[test],
                            interpolation_grid=L.lambda_values_)
    np.testing.assert_allclose(predictions[test], expected, rtol=1e-10)
