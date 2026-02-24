import numpy as np
import pytest
from glmnet.data import make_dataset
from glmnet.paths import LogNet, GaussNet, MultiGaussNet, MultiClassNet, FishNet

@pytest.mark.parametrize("n_samples,n_features,n_informative,snr,bias", [
    (50, 8, 4, 3, 0.0),
    (30, 5, 3, None, 1.5),
    (10, 10, 10, 1, -2.0),
])
def test_gaussian(n_samples, n_features, n_informative, snr, bias):
    if snr is not None:
        X, y, coef, intercept = make_dataset(GaussNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, snr=snr, bias=bias)
    else:
        X, y, coef, intercept = make_dataset(GaussNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, bias=bias)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert coef.shape == (n_features,)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.isscalar(intercept) or (isinstance(intercept, np.ndarray) and intercept.shape == ())

@pytest.mark.parametrize("n_samples,n_features,n_informative,n_targets,snr,bias", [
    (40, 6, 3, 2, 2, 0.0),
    (20, 4, 2, 3, None, [1.0, -1.0, 0.5]),
])
def test_multigaussian(n_samples, n_features, n_informative, n_targets, snr, bias):
    if snr is not None:
        X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets, snr=snr, bias=bias)
    else:
        X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets, bias=bias)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples, n_targets)
    assert coef.shape == (n_features, n_targets)
    assert np.issubdtype(y.dtype, np.floating)
    assert isinstance(intercept, np.ndarray) and intercept.shape == (n_targets,)

@pytest.mark.parametrize("n_samples,n_features,n_informative,snr,bias", [
    (60, 7, 3, 4, 0.0),
    (15, 5, 2, None, -1.0),
])
def test_binomial(n_samples, n_features, n_informative, snr, bias):
    X, y, coef, intercept = make_dataset(LogNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, snr=snr, bias=bias)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert coef.shape == (n_features,)
    assert set(np.unique(y)).issubset({0, 1})
    assert np.isscalar(intercept) or (isinstance(intercept, np.ndarray) and intercept.shape == ())

@pytest.mark.parametrize("n_samples,n_features,n_informative,n_classes,snr,bias", [
    (70, 5, 2, 4, 2, 0.0),
    (25, 6, 3, 3, None, [0.5, -0.5, 1.0]),
])
def test_multiclass(n_samples, n_features, n_informative, n_classes, snr, bias):
    X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_classes, snr=snr, bias=bias)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert coef.shape == (n_features, n_classes)
    assert set(np.unique(y)).issubset(set(range(n_classes)))
    assert isinstance(intercept, np.ndarray) and intercept.shape == (n_classes,)

@pytest.mark.parametrize("n_samples,n_features,n_informative,snr,bias", [
    (30, 4, 2, 1.5, 0.0),
    (12, 6, 3, None, 2.0),
])
def test_poisson(n_samples, n_features, n_informative, snr, bias):
    X, y, coef, intercept = make_dataset(FishNet, n_samples=n_samples, n_features=n_features, n_informative=n_informative, snr=snr, bias=bias)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert coef.shape == (n_features,)
    assert np.all(y >= 0)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.isscalar(intercept) or (isinstance(intercept, np.ndarray) and intercept.shape == ()) 
