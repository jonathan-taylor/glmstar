import numpy as np
import pandas as pd

import pytest

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glmnet.regularized_glm import (RegGLM,
                                    GaussianRegGLM,
                                    BinomialRegGLM)

# RegGLM and its Gaussian / binomial versions as sklearn estimators: they
# construct without arguments, clone, and tune in GridSearchCV

rng = np.random.default_rng(0)
n, p = 100, 5
X = rng.standard_normal((n, p))
y = X[:, 0] - X[:, 1] + rng.standard_normal(n)
y_binary = (y > 0).astype(int)


@pytest.mark.parametrize('klass', [RegGLM, GaussianRegGLM, BinomialRegGLM])
def test_clone(klass):

    est = klass()
    assert est.lambda_val == 0
    est.set_params(lambda_val=0.1, alpha=0.5)
    params = est.get_params()
    assert params['lambda_val'] == 0.1 and params['alpha'] == 0.5
    cloned = clone(est)
    assert cloned.get_params()['lambda_val'] == 0.1
    assert cloned.get_params()['alpha'] == 0.5


def test_gaussian_regglm():

    # GaussianRegGLM fits (its family check used a family set only by fit)
    fit = GaussianRegGLM(lambda_val=0.1).fit(X, y)
    ref = RegGLM(lambda_val=0.1).fit(X, y)
    np.testing.assert_allclose(fit.coef_, ref.coef_)
    np.testing.assert_allclose(fit.predict(X), ref.predict(X))
    # a DataFrame X
    X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    np.testing.assert_allclose(fit.predict(X_df), fit.predict(X))


def test_binomial_regglm():

    fit = BinomialRegGLM(lambda_val=0.01).fit(X, y_binary)
    prob = fit.predict_proba(X)
    np.testing.assert_allclose(prob.sum(1), 1)
    np.testing.assert_array_equal(fit.predict(X), fit.classes_[(prob[:, 1] > 0.5).astype(int)])


def test_gridsearch_pipeline():

    pipe = Pipeline([('s', StandardScaler()), ('r', RegGLM(alpha=0))])
    grid = GridSearchCV(pipe, {'r__lambda_val': [0.01, 0.1, 1]},
                        scoring='neg_mean_squared_error', cv=3)
    grid.fit(X, y)
    assert grid.best_params_['r__lambda_val'] in [0.01, 0.1, 1]


@pytest.mark.parametrize('klass, response, scoring',
                         [(GaussianRegGLM, y, 'neg_mean_squared_error'),
                          (BinomialRegGLM, y_binary, 'accuracy')])
def test_gridsearch(klass, response, scoring):

    grid = GridSearchCV(klass(), {'lambda_val': [0.01, 0.1], 'alpha': [0, 1]},
                        scoring=scoring, cv=3)
    grid.fit(X, response)
    assert set(grid.best_params_) == {'lambda_val', 'alpha'}
    grid.predict(X)
