# the classes of a binomial or multinomial response are learned at fit:
# scoring other data (e.g. a fold with one class) codes it by them

import numpy as np
import pandas as pd
import pytest

from glmnet.glm import BinomialGLM
from glmnet.regularized_glm import BinomialRegGLM
from glmnet.paths import LogNet, MultiClassNet

rng = np.random.default_rng(0)   # used only here, at import
n, p = 80, 4
X = rng.standard_normal((n, p))
labels = np.where(X[:, 0] + rng.standard_normal(n) > 0, 'b', 'a')
coded = (labels == 'b').astype(float)
# a subset with one class only
only_b = labels == 'b'


def _scores(fit, X, y):
    return fit.score_path(X, y).scores.to_numpy()


def test_score_path_one_class():
    by_label = LogNet().fit(X, labels)
    by_code = LogNet().fit(X, coded)
    np.testing.assert_allclose(by_label.coefs_, by_code.coefs_)
    # (successes, failures), R's cbind, are coded the same whatever the classes
    counts = np.column_stack([coded, 1 - coded])
    np.testing.assert_allclose(_scores(by_label, X[only_b], labels[only_b]),
                               _scores(by_code, X[only_b], counts[only_b]))
    # scoring does not change the classes
    assert list(by_label.classes_) == ['a', 'b']


def test_score_path_unseen_label():
    fit = LogNet().fit(X, labels)
    with pytest.raises(ValueError):
        fit.score_path(X, np.where(only_b, 'b', 'c'))


@pytest.mark.parametrize('est', [BinomialGLM, lambda: BinomialRegGLM(lambda_val=0.01)])
def test_glm_classes(est):
    fit = est().fit(X, labels)
    assert list(fit.classes_) == ['a', 'b']
    np.testing.assert_array_equal(fit.predict(X[only_b]),
                                  est().fit(X, labels).predict(X[only_b]))
    # a new fit learns its own classes
    fit.fit(X, np.where(only_b, 'y', 'x'))
    assert list(fit.classes_) == ['x', 'y']


def test_multiclass_fold_missing_class():
    classes = np.array(['a', 'b', 'c'])[np.random.default_rng(1).integers(0, 3, n)]
    onehot = np.stack([classes == c for c in 'abc'], axis=1).astype(float)
    by_label = MultiClassNet().fit(X, classes)
    by_count = MultiClassNet().fit(X, onehot)
    np.testing.assert_allclose(by_label.coefs_, by_count.coefs_)
    keep = classes != 'c'
    np.testing.assert_allclose(_scores(by_label, X[keep], classes[keep]),
                               _scores(by_count, X[keep], onehot[keep]))
    assert list(by_label.categories_) == ['a', 'b', 'c']


def test_lognet_refit_relearns():
    fit = LogNet().fit(X, labels)
    fit.fit(X, np.where(only_b, 'y', 'x'))
    assert list(fit.classes_) == ['x', 'y']
    # a DataFrame y, with the ids
    Y = pd.DataFrame({'y': labels, 'w': np.ones(n)})
    fit = LogNet(response_id='y', weight_id='w').fit(X, Y)
    counts = np.column_stack([coded, 1 - coded])
    np.testing.assert_allclose(_scores(fit, X[only_b], Y[only_b]),
                               _scores(LogNet().fit(X, coded), X[only_b], counts[only_b]))
