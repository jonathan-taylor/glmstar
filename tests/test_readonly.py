"""The C++ solvers accept read-only arrays for the inputs they only read.

A scipy sparse matrix shared with R by reticulate (a ``dgCMatrix``), or a
memory-mapped array, has read-only buffers. The bindings in ``src/*.cpp``
map these without copying, so they must not ask pybind11 for writeable
data.
"""
import functools

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import statsmodels.api as sm

import glmnet.elnet as elnet
from glmnet import GLMNet, LogNet, GaussNet, FishNet, MultiGaussNet, CoxNet
from glmnet.regularized_glm import RegGLM

rng = np.random.default_rng(0)
N, P = 150, 8
X_DENSE = rng.standard_normal((N, P))
X_DENSE[rng.uniform(size=X_DENSE.shape) < 0.6] = 0
ETA = X_DENSE[:, :3] @ np.array([1., -0.5, 0.5])
T = rng.exponential(np.exp(-ETA / 2))
C = rng.exponential(np.median(T) * 2, size=N)

RESPONSES = {
    'gaussian': ETA + rng.standard_normal(N),
    'binomial': rng.binomial(1, 1 / (1 + np.exp(-ETA))).astype(float),
    'poisson': rng.poisson(np.exp(ETA / 2)).astype(float),
    'multigaussian': np.column_stack([ETA, -ETA]) + rng.standard_normal((N, 2)),
}
COX_RESPONSE = pd.DataFrame({'event': np.minimum(T, C), 'status': (T <= C).astype(float)})

LAM = dict(lambda_values=np.array([0.5, 0.1, 0.02]))
CASES = {
    'GLMNet(Binomial)': (lambda: GLMNet(family=sm.families.Binomial(), **LAM), 'binomial'),
    'GLMNet(Gaussian)': (lambda: GLMNet(**LAM), 'gaussian'),
    'GLMNet(Poisson)': (lambda: GLMNet(family=sm.families.Poisson(), **LAM), 'poisson'),
    'RegGLM(Binomial)': (lambda: RegGLM(family=sm.families.Binomial(), lambda_val=0.05),
                         'binomial'),
    'LogNet': (lambda: LogNet(**LAM), 'binomial'),
    'GaussNet': (lambda: GaussNet(**LAM), 'gaussian'),
    'FishNet': (lambda: FishNet(**LAM), 'poisson'),
    'MultiGaussNet': (lambda: MultiGaussNet(**LAM), 'multigaussian'),
    'CoxNet': (lambda: CoxNet(**LAM), 'cox'),
}


def _readonly(a):
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


def _readonly_X(sparse):
    if sparse:
        X = sp.csc_matrix(X_DENSE)
        X.data, X.indices, X.indptr = map(_readonly, (X.data, X.indices, X.indptr))
        return X
    return _readonly(np.asfortranarray(X_DENSE))


def _response(kind):
    if kind == 'cox':
        return COX_RESPONSE
    return RESPONSES[kind]


def _readonly_response(kind):
    if kind == 'cox':
        return pd.DataFrame({k: _readonly(v) for k, v in COX_RESPONSE.items()})
    return _readonly(RESPONSES[kind])


def _coef(model):
    return np.asarray(model.coefs_ if hasattr(model, 'coefs_') else model.coef_)


def _X_arrays(X):
    return (X.data, X.indices, X.indptr) if sp.issparse(X) else (X,)


@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
@pytest.mark.parametrize('name', list(CASES))
def test_readonly_inputs(name, sparse):
    make, kind = CASES[name]
    X = sp.csc_matrix(X_DENSE) if sparse else np.asfortranarray(X_DENSE)
    fit = make().fit(X, _response(kind))

    X_ro, y_ro = _readonly_X(sparse), _readonly_response(kind)
    before = [a.copy() for a in _X_arrays(X_ro)]
    fit_ro = make().fit(X_ro, y_ro)

    np.testing.assert_array_equal(_coef(fit_ro), _coef(fit))
    for a, b in zip(_X_arrays(X_ro), before):
        assert not a.flags.writeable
        np.testing.assert_array_equal(a, b)


def _readonly_args(solver, names):
    # call the solver as the estimator would, with the inputs it only reads
    # handed over as read-only views
    @functools.wraps(solver)
    def wrapped(**args):
        for k in names:
            if k in args:
                v = np.asarray(args[k]).view()
                v.flags.writeable = False
                args[k] = v
        return solver(**args)
    return wrapped


@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
def test_readonly_wls_point_solver(sparse, monkeypatch):
    """The point solvers behind GLMNet / RegGLM only read x, v, xm, xs, ju, vp, cl."""
    names = ['x', 'x_data_array', 'x_indices_array', 'x_indptr_array',
             'xm', 'xs', 'v', 'ju', 'vp', 'cl']
    make = lambda: GLMNet(family=sm.families.Binomial(), **LAM)
    y = RESPONSES['binomial']
    X = sp.csc_matrix(X_DENSE) if sparse else X_DENSE
    fit = make().fit(X, y)

    monkeypatch.setattr(elnet, 'sparse_wls', _readonly_args(elnet.sparse_wls, names))
    monkeypatch.setattr(elnet, 'dense_wls', _readonly_args(elnet.dense_wls, names))
    fit_ro = make().fit(X, y)
    np.testing.assert_array_equal(fit_ro.coefs_, fit.coefs_)


@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
@pytest.mark.parametrize('name', ['LogNet', 'GaussNet', 'FishNet', 'MultiGaussNet', 'CoxNet'])
def test_readonly_path_solver(name, sparse, monkeypatch):
    """The path solvers only read the sparse X, the penalty factors, the
    lambda values and the survival data (the dense X, y and weights are
    standardized in place by the solvers, so the estimators copy them)."""
    make, kind = CASES[name]
    X = sp.csc_matrix(X_DENSE) if sparse else X_DENSE
    fit = make().fit(X, _response(kind))

    cls = type(fit)
    names = ['x_data_array', 'x_indices_array', 'x_indptr_array',
             'jd', 'vp', 'ulam', 'start', 'stop', 'status', 'strata']
    if name == 'CoxNet':
        names.append('w')
    solver = '_sparse' if sparse else '_dense'
    monkeypatch.setattr(cls, solver, staticmethod(_readonly_args(getattr(cls, solver), names)))
    fit_ro = make().fit(X, _response(kind))
    np.testing.assert_array_equal(fit_ro.coefs_, fit.coefs_)
