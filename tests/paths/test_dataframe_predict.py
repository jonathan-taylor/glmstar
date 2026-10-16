import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm

import pytest

from glmnet import (GLMNet,
                    GaussNet,
                    LogNet,
                    FishNet,
                    MultiGaussNet,
                    MultiClassNet)

# fit accepts a DataFrame X; predict (and so cross_validation_path,
# cv_predict and score_path) should too, giving the same as for the array

rng = np.random.default_rng(0)
n, p = 80, 5
X = rng.standard_normal((n, p))
X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
y = X[:, 0] - X[:, 1] + rng.standard_normal(n)

cases = {'GaussNet': (GaussNet, y),
         'LogNet': (LogNet, y > 0),
         'FishNet': (FishNet, rng.poisson(np.exp(X[:, 0] / 2))),
         'MultiGaussNet': (MultiGaussNet, np.column_stack([y, -y + rng.standard_normal(n)])),
         'MultiClassNet': (MultiClassNet, rng.integers(0, 3, n)),
         'GLMNet': (GLMNet, y),
         'GLMNet-binomial': (lambda: GLMNet(family=sm.families.Binomial()),
                             (y > 0).astype(float))}


@pytest.mark.parametrize('name', sorted(cases))
def test_predict_dataframe(name):

    klass, response = cases[name]
    fit = klass().fit(X_df, response)

    np.testing.assert_allclose(fit.predict(X_df), fit.predict(X))
    lam = fit.lambda_values_[[2, 5]]
    np.testing.assert_allclose(fit.predict(X_df, interpolation_grid=lam),
                               fit.predict(X, interpolation_grid=lam))
    np.testing.assert_allclose(fit.predict(X_df, prediction_type='link'),
                               fit.predict(X, prediction_type='link'))
    # sparse X is still accepted
    np.testing.assert_allclose(fit.predict(scipy.sparse.csc_array(X)),
                               fit.predict(X))


@pytest.mark.parametrize('name', ['GaussNet', 'LogNet', 'GLMNet'])
def test_cross_validation_dataframe(name):

    klass, response = cases[name]
    fit_df = klass().fit(X_df, response)
    fit = klass().fit(X, response)

    # the same folds for both fits
    folds = np.arange(n) % 4
    cv = [(np.nonzero(folds != i)[0], np.nonzero(folds == i)[0]) for i in range(4)]
    preds_df, path_df = fit_df.cross_validation_path(X_df, response, cv=cv)
    preds, path = fit.cross_validation_path(X, response, cv=cv)

    np.testing.assert_allclose(preds_df, preds)
    pd.testing.assert_frame_equal(path_df.scores, path.scores[path_df.scores.columns])
    np.testing.assert_allclose(fit_df.cv_predict(X_df), fit.cv_predict(X))

    score_df = fit_df.score_path(X_df, response)
    score = fit.score_path(X, response)
    pd.testing.assert_frame_equal(score_df.scores, score.scores[score_df.scores.columns])
