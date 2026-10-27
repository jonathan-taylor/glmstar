import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import pytest

from glmnet.cox import CoxLM, CoxFamily, CoxFamilySpec
from modelmatrix import Surv
from glmnet.glm import GLM, GLMControl

TIGHT_LIFELINES = {'precision': 1e-12}

# CoxLM's Newton steps use the full information: at the default control the
# fit is the maximum of the partial likelihood (IRLS with the diagonal of the
# Hessian stopped at mxitnr far from it, without a warning)


def _data(seed=0, n=85, p=8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # correlated columns, which slow IRLS down most
    X[:, 1:] += X[:, :1]
    beta = np.linspace(-1, 1, p)
    t = rng.exponential(np.exp(-X @ beta / 2))
    c = rng.exponential(2, n)
    Y = pd.DataFrame({'time': np.round(np.minimum(t, c), 1) + 0.05,    # with ties
                      'status': (t <= c).astype(float)})
    return X, Y


def _score(fit, X, Y):
    # the score of the partial likelihood at the fit, X^T (minus half the deviance's gradient)
    family = CoxFamilySpec(event_data=Y, tie_breaking=fit.family.tie_breaking,
                           event_id='time', status_id='status')
    result = family._coxdev(X @ fit.coef_, np.ones(X.shape[0]))
    return -X.T @ result.gradient / 2


@pytest.mark.parametrize('ties', ['breslow', 'efron'])
def test_coxlm_converges(ties):
    X, Y = _data()
    family = CoxFamily(tie_breaking=ties)
    outcome = Surv('time', 'status')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fit = CoxLM(family=family, outcome=outcome, summarize=True).fit(X, Y)
    tight = CoxLM(family=family, outcome=outcome,
                  control=GLMControl(epsnr=1e-14, mxitnr=50)).fit(X, Y)
    np.testing.assert_allclose(fit.coef_, tight.coef_, atol=1e-6)
    np.testing.assert_allclose(_score(fit, X, Y), 0, atol=1e-6)


def test_coxlm_lifelines():
    lifelines = pytest.importorskip('lifelines')
    X, Y = _data(seed=1)
    fit = CoxLM(family=CoxFamily(event_id='time', status_id='status',
                                 tie_breaking='efron')).fit(X, Y)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])]).assign(**Y)
    cph = lifelines.CoxPHFitter().fit(df, 'time', 'status', fit_options=TIGHT_LIFELINES)
    np.testing.assert_allclose(fit.coef_, cph.params_.values, atol=1e-6)
    np.testing.assert_allclose(fit.loglike_, cph.log_likelihood_, rtol=1e-8)


def test_braincancer():
    ISLP = pytest.importorskip('ISLP')
    from ISLP.models import ModelSpec
    lifelines = pytest.importorskip('lifelines')

    cleaned = ISLP.load_data('BrainCancer').dropna()
    X = ModelSpec(cleaned.columns.drop(['time', 'status']),
                  intercept=False).fit_transform(cleaned)
    Y = cleaned[['time', 'status']]
    fit = CoxLM(family=CoxFamily(event_id='time', status_id='status',
                                 tie_breaking='efron'),
                summarize=True).fit(X, Y)
    cph = lifelines.CoxPHFitter().fit(X.assign(**Y), 'time', 'status',
                                      fit_options=TIGHT_LIFELINES)
    np.testing.assert_allclose(fit.coef_, cph.params_.values, atol=1e-6)
    np.testing.assert_allclose(fit.loglike_, cph.log_likelihood_, rtol=1e-8)


def test_mxitnr_warning():
    X, Y = _data()
    family = CoxFamily(event_id='time', status_id='status')
    with pytest.warns(UserWarning, match='did not converge'):
        CoxLM(family=family, control=GLMControl(mxitnr=1)).fit(X, Y)

    rng = np.random.default_rng(0)
    y = rng.poisson(np.exp(X[:, 0] / 2))
    with pytest.warns(UserWarning, match='did not converge'):
        GLM(family=sm.families.Poisson(), control=GLMControl(mxitnr=1)).fit(X, y)
