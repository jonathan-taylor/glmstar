"""
A sweep over the estimators: each constructs with defaults and clones;
fits and predicts with a DataFrame and a sparse X as with an array; the
path estimators cross-validate with a DataFrame X; the fixed-lambda
estimators tune lambda in GridSearchCV; the fixed-lambda fit from
`get_fixed_lambda` solves the path's problem for alpha in {0, 0.5, 1}; and
the unpenalized fits converge at the default control (and warn when not).
"""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm

import pytest

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from glmnet import (GLM,
                    GLMNet,
                    CoxNetIRLS,
                    GaussNet,
                    LogNet,
                    FishNet,
                    MultiGaussNet,
                    MultiClassNet,
                    CoxNet)
from glmnet.glm import GaussianGLM, BinomialGLM, GLMControl
from glmnet.regularized_glm import RegGLM, GaussianRegGLM, BinomialRegGLM
from glmnet.cox import CoxLM, RegCoxLM, CoxFamily

rng = np.random.default_rng(0)
n, p = 80, 4
X = rng.standard_normal((n, p))
X_df = pd.DataFrame(X, columns=list('abcd'))
X_sp = scipy.sparse.csc_array(X)
eta = X @ [1, -1, 0.5, 0]
Y = {'gaussian': eta + rng.standard_normal(n),
     'binomial': rng.binomial(1, 1 / (1 + np.exp(-eta))),
     'poisson': rng.poisson(np.exp(eta / 2)),
     'mgaussian': np.column_stack([eta, -eta]) + rng.standard_normal((n, 2)),
     'multinomial': rng.integers(0, 3, n)}
T = rng.exponential(np.exp(-eta))
C = rng.exponential(1.5, n)
Y['cox'] = pd.DataFrame({'event': np.round(np.minimum(T, C), 1) + 0.05,
                         'status': (T <= C).astype(float)})
COX = dict(family=CoxFamily(tie_breaking='efron'))
POISSON = dict(family=sm.families.Poisson())

# name: (class, keyword arguments, response)
ESTIMATORS = {'GLM': (GLM, {}, 'gaussian'),
              'GLM-poisson': (GLM, POISSON, 'poisson'),
              'GaussianGLM': (GaussianGLM, {}, 'gaussian'),
              'BinomialGLM': (BinomialGLM, {}, 'binomial'),
              'CoxLM': (CoxLM, COX, 'cox'),
              'RegGLM': (RegGLM, dict(lambda_val=0.1), 'gaussian'),
              'RegGLM-poisson': (RegGLM, dict(lambda_val=0.05, **POISSON), 'poisson'),
              'GaussianRegGLM': (GaussianRegGLM, dict(lambda_val=0.1), 'gaussian'),
              'BinomialRegGLM': (BinomialRegGLM, dict(lambda_val=0.05), 'binomial'),
              'RegCoxLM': (RegCoxLM, dict(lambda_val=0.05, **COX), 'cox'),
              'GLMNet': (GLMNet, {}, 'gaussian'),
              'CoxNetIRLS': (CoxNetIRLS, COX, 'cox'),
              'GaussNet': (GaussNet, {}, 'gaussian'),
              'LogNet': (LogNet, {}, 'binomial'),
              'FishNet': (FishNet, {}, 'poisson'),
              'MultiGaussNet': (MultiGaussNet, {}, 'mgaussian'),
              'MultiClassNet': (MultiClassNet, {}, 'multinomial'),
              'CoxNet': (CoxNet, COX, 'cox')}
PATHS = [name for name, (klass, _, _) in ESTIMATORS.items()
         if hasattr(klass, 'cross_validation_path')]
FIXED = ['RegGLM', 'RegGLM-poisson', 'GaussianRegGLM', 'BinomialRegGLM', 'RegCoxLM']


def _make(name):
    klass, kwargs, response = ESTIMATORS[name]
    return klass(**kwargs), Y[response]


@pytest.mark.parametrize('name', sorted(ESTIMATORS))
def test_defaults_clone(name):
    klass, kwargs, _ = ESTIMATORS[name]
    klass()
    est = klass(**kwargs)
    params = est.get_params()
    cloned = clone(est)
    assert set(cloned.get_params()) == set(params)
    cloned.set_params(**params)
    for key in kwargs:
        assert cloned.get_params()[key] == kwargs[key]


@pytest.mark.parametrize('name', sorted(ESTIMATORS))
def test_dataframe_sparse(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        est, y = _make(name)
        dense = clone(est).fit(X, y).predict(X)
        fit_df = clone(est).fit(X_df, y)
        np.testing.assert_allclose(fit_df.predict(X_df), dense)
        np.testing.assert_allclose(fit_df.predict(X), dense)
        fit_sp = clone(est).fit(X_sp, y)
        np.testing.assert_allclose(fit_sp.predict(X_sp), dense, atol=1e-4)


@pytest.mark.parametrize('name', sorted(PATHS))
def test_cross_validation_dataframe(name):
    est, y = _make(name)
    fit = est.fit(X_df, y)
    folds = np.arange(n) % 3
    cv = [(np.nonzero(folds != i)[0], np.nonzero(folds == i)[0]) for i in range(3)]
    predictions, _ = fit.cross_validation_path(X_df, y, cv=cv)
    predictions_array, _ = clone(est).fit(X, y).cross_validation_path(X, y, cv=cv)
    np.testing.assert_allclose(predictions, predictions_array, atol=1e-10)
    fit.cv_predict(X_df)


@pytest.mark.parametrize('name', FIXED)
def test_gridsearch(name):
    est, y = _make(name)
    scoring = {'GaussianRegGLM': 'neg_mean_squared_error',
               'BinomialRegGLM': 'accuracy'}.get(name)
    grid = GridSearchCV(est, {'lambda_val': [0.01, 0.1], 'alpha': [0, 1]},
                        scoring=scoring, cv=3, error_score='raise').fit(X, y)
    assert np.all(np.isfinite(grid.cv_results_['mean_test_score']))
    grid.predict(X)


# the single lambda fits iterate to their own convergence criteria
TOL = {'gaussian': 1e-5, 'binomial': 1e-4, 'poisson': 1e-4, 'cox': 2e-3}


@pytest.mark.parametrize('family, path_class', [('gaussian', GaussNet),
                                          ('binomial', LogNet),
                                          ('poisson', FishNet),
                                          ('cox', lambda **kw: CoxNet(**COX, **kw))])
@pytest.mark.parametrize('mixing', [0, 0.5, 1])
def test_fixed_lambda_scale(family, path_class, mixing):
    y = Y[family]
    L = path_class(alpha=mixing).fit(X, y)
    k = 10
    estimator, _ = L.get_fixed_lambda(L.lambda_values_[k])
    estimator.fit(X, y)
    np.testing.assert_allclose(estimator.coef_, L.coefs_[k], atol=TOL[family])
    if family != 'gaussian':
        # glmnet rescales the response only for the gaussian: the fixed
        # lambda estimator is the same as the path's
        assert estimator.lambda_val == L.lambda_values_[k]
        assert estimator.alpha == mixing


@pytest.mark.parametrize('family, response', [(sm.families.Binomial(), 'binomial'),
                                              (sm.families.Binomial(sm.families.links.Probit()), 'binomial'),
                                              (sm.families.Poisson(), 'poisson'),
                                              (sm.families.Gaussian(), 'gaussian')])
def test_unpenalized_convergence(family, response):
    y = Y[response]
    reference = sm.GLM(y, sm.add_constant(X), family=family).fit(tol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fit = GLM(family=family).fit(X, y)
    np.testing.assert_allclose(fit.params_.values, reference.params, atol=1e-5)
    if response != 'gaussian':
        # (the gaussian converges in one step)
        with pytest.warns(UserWarning, match='did not converge'):
            GLM(family=family, control=GLMControl(mxitnr=1)).fit(X, y)
