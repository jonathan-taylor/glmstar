import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import statsmodels.api as sm
from scipy.special import expit

from glmnet import GLMNet, GLM, CoxNetIRLS, GaussNet, LogNet, FishNet, CoxNet
from glmnet.glmnet import GLMNetControl
from glmnet.regularized_glm import RegGLM
from glmnet.cox import CoxFamily

N, P = 100, 8

FAMILIES = {'gaussian': sm.families.Gaussian(),
            'binomial': sm.families.Binomial(),
            'poisson': sm.families.Poisson(),
            'probit': sm.families.Binomial(link=sm.families.links.Probit())}


def _sparse_frame(X):
    # a pandas DataFrame of sparse columns (zero fill value)
    return pd.DataFrame({f'X{j}': pd.arrays.SparseArray(X[:, j], fill_value=0.)
                         for j in range(X.shape[1])})


FORMATS = {'csc_matrix': scipy.sparse.csc_matrix,
           'csr_matrix': scipy.sparse.csr_matrix,
           'csc_array': scipy.sparse.csc_array,
           'csr_array': scipy.sparse.csr_array,
           'sparse_frame': _sparse_frame}


def get_data(family, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    X[rng.uniform(size=X.shape) < 0.6] = 0
    eta = 0.5 * X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family == 'gaussian':
        y = eta + rng.standard_normal(N)
    elif family in ['binomial', 'probit']:
        y = rng.binomial(1, expit(eta)).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    D = pd.DataFrame({'Y': y,
                      'w': rng.uniform(0.5, 2, N),
                      'o': 0.2 * rng.standard_normal(N)})
    return X, D


def _glmnet(family, standardize, fit_intercept, use_weights, use_offset):
    return GLMNet(family=FAMILIES[family],
                  response_id='Y',
                  weight_id='w' if use_weights else None,
                  offset_id='o' if use_offset else None,
                  standardize=standardize,
                  fit_intercept=fit_intercept,
                  nlambda=20,
                  control=GLMNetControl(thresh=1e-12))


@pytest.mark.parametrize('family', list(FAMILIES))
@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_array', 'sparse_frame'])
def test_glmnet_sparse_matches_dense(family, fmt, standardize, fit_intercept, use_weights, use_offset):
    X, D = get_data(family)
    dense = _glmnet(family, standardize, fit_intercept, use_weights, use_offset).fit(X, D)
    sparse = _glmnet(family, standardize, fit_intercept, use_weights, use_offset).fit(FORMATS[fmt](X), D)
    np.testing.assert_allclose(sparse.lambda_values_, dense.lambda_values_, rtol=1e-10)
    np.testing.assert_allclose(sparse.coefs_, dense.coefs_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sparse.intercepts_, dense.intercepts_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('fmt', list(FORMATS))
def test_glmnet_sparse_formats(fmt):
    X, D = get_data('binomial')
    dense = _glmnet('binomial', True, True, True, False).fit(X, D)
    sparse = _glmnet('binomial', True, True, True, False).fit(FORMATS[fmt](X), D)
    np.testing.assert_allclose(sparse.coefs_, dense.coefs_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_array'])
def test_glmnet_sparse_predict_cv_score(fmt):
    X, D = get_data('poisson')
    Xs = FORMATS[fmt](X)
    dense = _glmnet('poisson', True, True, True, True).fit(X, D)
    sparse = _glmnet('poisson', True, True, True, True).fit(Xs, D)

    for prediction_type in ['link', 'response']:
        np.testing.assert_allclose(sparse.predict(Xs, prediction_type=prediction_type),
                                   dense.predict(X, prediction_type=prediction_type),
                                   rtol=1e-6, atol=1e-8)

    cv = [(np.arange(N) % 4 != k, np.arange(N) % 4 == k) for k in range(4)]
    cv = [(np.nonzero(tr)[0], np.nonzero(te)[0]) for tr, te in cv]
    pred_d, path_d = dense.cross_validation_path(X, D, cv=cv)
    pred_s, path_s = sparse.cross_validation_path(Xs, D, cv=cv)
    np.testing.assert_allclose(pred_s, pred_d, rtol=1e-6, atol=1e-8)
    # the order of the score columns is not fixed (check_like would also
    # realign the rows on lambda values that differ by roundoff)
    def same_scores(a, b):
        pd.testing.assert_frame_equal(a[b.columns], b, rtol=1e-6)

    same_scores(path_s.scores, path_d.scores)

    same_scores(sparse.score_path(Xs, D).scores, dense.score_path(X, D).scores)


@pytest.mark.parametrize('family', list(FAMILIES))
@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_array'])
def test_glm_sparse_matches_dense(family, fmt):
    X, D = get_data(family)
    y, w = D['Y'].values, D['w'].values
    dense = GLM(family=FAMILIES[family]).fit(X, y, sample_weight=w)
    sparse = GLM(family=FAMILIES[family]).fit(FORMATS[fmt](X), y, sample_weight=w)
    np.testing.assert_allclose(sparse.coef_, dense.coef_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sparse.intercept_, dense.intercept_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('family', list(FAMILIES))
@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_array'])
def test_regglm_sparse_matches_dense(family, fmt, standardize):
    X, D = get_data(family)
    y = D['Y'].values

    def fit(X_):
        return RegGLM(family=FAMILIES[family], lambda_val=0.02, alpha=0.5,
                      standardize=standardize).fit(X_, y)

    dense, sparse = fit(X), fit(FORMATS[fmt](X))
    np.testing.assert_allclose(sparse.coef_, dense.coef_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sparse.intercept_, dense.intercept_, rtol=1e-6, atol=1e-8)


def _cox_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    X[rng.uniform(size=X.shape) < 0.6] = 0
    eta = 0.5 * X[:, 0] - 0.5 * X[:, 1]
    D = pd.DataFrame({'stop': rng.exponential(np.exp(-eta)),
                      'status': rng.binomial(1, 0.8, N)})
    return X, D


@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_array'])
def test_coxnet_irls_sparse_matches_dense(fmt, standardize):
    X, D = _cox_data()

    def fit(X_):
        return CoxNetIRLS(family=CoxFamily(event_id='stop', status_id='status'),
                          standardize=standardize, nlambda=15,
                          control=GLMNetControl(thresh=1e-12)).fit(X_, D)

    dense, sparse = fit(X), fit(FORMATS[fmt](X))
    np.testing.assert_allclose(sparse.coefs_, dense.coefs_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('fmt', ['csc_matrix', 'csr_matrix', 'csr_array'])
@pytest.mark.parametrize('estimator', ['gaussian', 'binomial', 'poisson', 'cox'])
def test_cpp_paths_sparse_formats(fmt, estimator):
    if estimator == 'cox':
        X, D = _cox_data()
        make = lambda: CoxNet(family=CoxFamily(event_id='stop', status_id='status'), nlambda=20)
    else:
        X, D = get_data(estimator)
        cls = {'gaussian': GaussNet, 'binomial': LogNet, 'poisson': FishNet}[estimator]
        make = lambda: cls(response_id='Y', weight_id='w', nlambda=20)
    dense = make().fit(X, D)
    sparse = make().fit(FORMATS[fmt](X), D)
    np.testing.assert_allclose(sparse.coefs_, dense.coefs_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sparse.predict(FORMATS[fmt](X)), dense.predict(X), rtol=1e-6, atol=1e-8)
