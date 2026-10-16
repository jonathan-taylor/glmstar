import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import cross_validate, KFold

from glmnet import GaussNet, LogNet, CVNet

rng = np.random.default_rng(0)
n, p = 120, 6
X = rng.standard_normal((n, p))
y = X[:, 0] - X[:, 1] + rng.standard_normal(n)


def test_cvnet_matches_path():
    cv = KFold(5, shuffle=True, random_state=0)
    for which in ['1se', 'best']:
        model = CVNet(GaussNet(), cv=cv, which=which).fit(X, y)
        path = GaussNet().fit(X, y)
        path.cross_validation_path(X, y, cv=cv)
        lam, _ = path.cv_choice(which=which)
        assert model.lambda_ == lam
        np.testing.assert_allclose(model.predict(X), path.cv_predict(X, which=which))
        np.testing.assert_allclose(model.coef_, path.cv_coefs(which=which)[0])


def test_cvnet_nested():
    # nested cross-validation in one call, with a DataFrame X
    X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    result = cross_validate(CVNet(GaussNet(), cv=5), X_df, y, cv=3,
                            scoring='neg_mean_squared_error')
    assert np.all(np.isfinite(result['test_score']))
    yb = (y > 0).astype(int)
    model = CVNet(LogNet(), cv=5, prediction_type='class')
    result = cross_validate(model, X, yb, cv=3, scoring='accuracy')
    assert np.all(result['test_score'] > 0.5)
    clone(model)
