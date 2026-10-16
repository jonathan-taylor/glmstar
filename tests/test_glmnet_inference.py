import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.special import expit
import statsmodels.genmod.families as smf

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl

from glmnet.inference import problem_from_solution, glmstar_problem
from glmnet.inference import glm_inference_problem, glmstar_inference_problem
from glmnet.inference.glmnet_inference import _newton_step

N, P = 300, 10
SM_FAMILIES = {'gaussian': smf.Gaussian(), 'binomial': smf.Binomial(), 'poisson': smf.Poisson()}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, P))
    eta = X[:, :3] @ np.array([0.8, -0.6, 0.5])
    df = pd.DataFrame({'gaussian': eta + rng.standard_normal(N),
                       'binomial': rng.binomial(1, expit(eta)).astype(float),
                       'poisson': rng.poisson(np.exp(0.3 * eta)).astype(float),
                       'w': rng.uniform(0.5, 2, N),
                       'o': rng.normal(0, 0.2, N)})
    rows = np.sort(rng.choice(N, 200, replace=False))
    return X, df, rows


def _fit(X, df, family, **opts):
    G = GLMNet(family=SM_FAMILIES[family], response_id=family, nlambda=20,
               control=GLMNetControl(thresh=1e-14, fdev=0), **opts)
    G.fit(X, df)
    return G


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson'])
def test_no_randomization(data, family):
    X, df, _ = data
    G = _fit(X, df, family)
    lam = G.lambda_values_[10]
    li = glmstar_inference_problem(G, X, df, lambda_val=lam, dispersion=2.)
    prob = glmstar_problem(G, X, df, lambda_val=lam)
    np.testing.assert_allclose(li.Z_full, -prob.G_hat + prob.Q_hat @ prob.beta_hat)
    np.testing.assert_allclose(li.Sigma, 2 * prob.Q_hat / N)
    assert li.scalar_noise == 0
    assert li.dispersion == 2.
    assert li.problem is not None
    args = li.lasso_args()
    np.testing.assert_allclose(args['Z_full'], li.Z_full)
    assert args['scalar_noise'] == 0 and args['Sigma_noise'] is None
    assert set(args) == {'beta_hat', 'G_hat', 'Q_hat', 'D', 'L', 'U',
                         'Z_full', 'Sigma', 'Sigma_noise', 'scalar_noise'}


def test_carving_newton_step(data):
    # Z_full = -G_full + Q b at one full-data Newton step on the active set
    X, df, rows = data
    y = df['binomial'].values
    G = _fit(X[rows], df.iloc[rows], 'binomial')
    lam = G.lambda_values_[10]
    li = glmstar_inference_problem(G, X, df, lambda_val=lam, selection_rows=rows)
    np.testing.assert_allclose(li.scalar_noise, (1 - 2 / 3) / (2 / 3))
    prob = glmstar_problem(G, X[rows], df.iloc[rows], lambda_val=lam)
    X1 = np.column_stack([np.ones(N), X])
    b = prob.beta_hat.copy()
    A = np.union1d([0], np.nonzero(b)[0])
    mu = expit(X1 @ b)
    b[A] -= np.linalg.solve(X1[:, A].T @ (X1[:, A] * (mu * (1 - mu))[:, None]) / N, X1[:, A].T @ (mu - y) / N)
    G_full = X1.T @ (expit(X1 @ b) - y) / N
    np.testing.assert_allclose(li.Z_full, -G_full + prob.Q_hat @ b, atol=1e-10)
    np.testing.assert_allclose(li.Sigma, prob.Q_hat / N)
    # boolean mask
    mask = np.zeros(N, bool)
    mask[rows] = True
    np.testing.assert_allclose(glmstar_inference_problem(G, X, df, lambda_val=lam, selection_rows=mask).Z_full, li.Z_full)


def test_weights_offset_ridge(data):
    # glmnet_inference from a GLMProblem, with weights, offset and a ridge term
    X, df, rows = data
    y, w, o = df['poisson'].values, df['w'].values, df['o'].values
    lam, alpha = 0.05, 0.5
    # any coefficients will do for the algebra
    coef = np.r_[0.25, -0.2, 0.1, np.zeros(P - 3)]
    prob = problem_from_solution(X[rows], y[rows], coef, 0.1, lam, family='poisson', weights=w[rows],
                          offset=o[rows], alpha=alpha)
    li = glm_inference_problem(prob, X, y, 'poisson', weights=w, offset=o, selection_rows=rows)
    np.testing.assert_allclose(li.scalar_noise, (w.sum() - w[rows].sum()) / w[rows].sum())
    X1 = np.column_stack([np.ones(N), X])
    wn = w / w.sum()
    b = prob.beta_hat.copy()
    A = np.nonzero(b)[0]
    mu = np.exp(o + X1 @ b)
    b[A] -= np.linalg.solve(X1[:, A].T @ (X1[:, A] * (wn * mu)[:, None]), X1[:, A].T @ (wn * (mu - y)))
    wr = w[rows] / w[rows].sum()
    scale = np.sqrt(wr @ X[rows]**2 - (wr @ X[rows])**2)
    ridge = np.r_[0., lam * (1 - alpha) * scale**2]
    G_full = X1.T @ (wn * (np.exp(o + X1 @ b) - y)) + ridge * b
    np.testing.assert_allclose(li.Z_full, -G_full + prob.Q_hat @ b, atol=1e-10)
    # the ridge term is not random: Sigma is the information without it
    np.testing.assert_allclose(prob.ridge, ridge)
    np.testing.assert_allclose(prob.Q_hat - prob.information(), np.diag(ridge), atol=1e-14)
    np.testing.assert_allclose(li.Sigma, prob.information() / w.sum())


def test_elastic_net_operator_information(data):
    # information() for the matrix-free Hessian matches the dense one
    X, df, rows = data
    y = df['binomial'].values
    coef = np.r_[0.25, -0.2, 0.1, np.zeros(P - 3)]
    kw = dict(family='binomial', alpha=0.3)
    dense = problem_from_solution(X, y, coef, 0.1, 0.05, **kw)
    op = problem_from_solution(X, y, coef, 0.1, 0.05, hessian='operator', **kw)
    np.testing.assert_allclose(op.information() @ np.eye(P + 1), dense.information(), atol=1e-12)
    assert np.all(dense.ridge[1:] > 0) and dense.ridge[0] == 0
    # with alpha = 1 the information is Q_hat
    lasso = problem_from_solution(X, y, coef, 0.1, 0.05, family='binomial')
    assert lasso.information() is lasso.Q_hat


def _dense(A, k):
    return A @ np.eye(k) if isinstance(A, LinearOperator) else np.asarray(A)


def _psd_form(P, fit_intercept=True, seed=0):
    # a general (not diagonal) quadratic form on the coefficients, 0 for the intercept
    A = np.random.default_rng(seed).standard_normal((P, P))
    R = np.zeros((P + fit_intercept,) * 2)
    R[fit_intercept:, fit_intercept:] = 0.1 * A @ A.T / P
    return R


def test_ridge_forms(data):
    # the quadratic part of the penalty as a float, a vector (with or without the
    # intercept), or a dense, sparse or matrix-free quadratic form
    X, df, _ = data
    y = df['binomial'].values
    coef = np.r_[0.25, -0.2, 0.1, np.zeros(P - 3)]
    k = P + 1
    beta = np.random.default_rng(1).standard_normal(k)
    for hessian in ['dense', 'operator']:
        base = problem_from_solution(X, y, coef, 0.1, 0.05, family='binomial', hessian=hessian)
        Q = _dense(base.Q_hat, k)

        # diagonal: a float is c * I on the coefficients, the intercept unpenalized
        c = 0.3
        diagonal = np.r_[0., np.full(P, c)]
        for ridge in [c, diagonal, diagonal[1:], np.diag(diagonal),
                      sp.diags(diagonal).tocsr(), aslinearoperator(np.diag(diagonal))]:
            prob = dataclasses.replace(base, ridge=ridge)
            np.testing.assert_allclose(_dense(prob.information(), k), Q - np.diag(diagonal),
                                       atol=1e-14)
            np.testing.assert_allclose(prob.ridge_gradient(beta), diagonal * beta, atol=1e-14)

        # general
        R = _psd_form(P)
        for ridge in [R, sp.csr_matrix(R), aslinearoperator(R)]:
            prob = dataclasses.replace(base, ridge=ridge)
            np.testing.assert_allclose(_dense(prob.information(), k), Q - R, atol=1e-14)
            np.testing.assert_allclose(prob.ridge_gradient(beta), R @ beta, atol=1e-14)

    # zero is no quadratic term
    assert dataclasses.replace(base, ridge=0.).information() is base.Q_hat
    assert dataclasses.replace(base, ridge=np.zeros((k, k))).ridge_form() is None

    # without an intercept a float penalizes every coordinate
    no_int = problem_from_solution(X, y, coef, 0., 0.05, family='binomial', fit_intercept=False)
    prob = dataclasses.replace(no_int, ridge=c)
    np.testing.assert_allclose(prob.ridge_gradient(beta[1:]), c * beta[1:])

    for bad in [np.ones(P - 1), np.ones((P, P)), sp.eye(P)]:
        with pytest.raises(ValueError, match='ridge'):
            dataclasses.replace(base, ridge=bad).ridge_form()
    with pytest.raises(ValueError, match='ridge'):
        dataclasses.replace(no_int, ridge=np.ones(P - 1)).ridge_form()


def test_carving_general_ridge(data):
    # with held-out rows, Z_full has the gradient R b_tilde of a general quadratic form and
    # Sigma the information Q_hat - R
    X, df, rows = data
    y = df['binomial'].values
    coef = np.r_[0.25, -0.2, 0.1, np.zeros(P - 3)]
    lasso = problem_from_solution(X[rows], y[rows], coef, 0.1, 0.05, family='binomial')
    R = _psd_form(P)
    prob = dataclasses.replace(lasso, ridge=R)
    li = glm_inference_problem(prob, X, y, 'binomial', selection_rows=rows)
    li_lasso = glm_inference_problem(lasso, X, y, 'binomial', selection_rows=rows)
    b_tilde = _newton_step(X, y, lasso.beta_hat, 'binomial', np.ones(N), np.zeros(N), True)
    np.testing.assert_allclose(li.Z_full, li_lasso.Z_full - R @ b_tilde, atol=1e-12)
    np.testing.assert_allclose(li.Sigma, (lasso.Q_hat - R) / N, atol=1e-14)


def test_gaussian_dispersion(data):
    X, df, _ = data
    y = df['gaussian'].values
    G = _fit(X, df, 'gaussian')
    lam = G.lambda_values_[10]
    li = glmstar_inference_problem(G, X, df, lambda_val=lam)
    prob = glmstar_problem(G, X, df, lambda_val=lam)
    X1 = np.column_stack([np.ones(N), X])
    resid = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    np.testing.assert_allclose(li.Sigma, resid @ resid / (N - P - 1) * prob.Q_hat / N)
    with pytest.raises(ValueError, match='dispersion'):
        glm_inference_problem(prob, X[:5], y[:5], 'gaussian')


def test_operator_matches_dense(data):
    X, df, rows = data
    G = _fit(X[rows], df.iloc[rows], 'binomial')
    lam = G.lambda_values_[10]
    dense = glmstar_inference_problem(G, X, df, lambda_val=lam, selection_rows=rows)
    op = glmstar_inference_problem(G, X, df, lambda_val=lam, selection_rows=rows, hessian='operator')
    np.testing.assert_allclose(op.Z_full, dense.Z_full, atol=1e-10)
    np.testing.assert_allclose(op.Sigma @ np.eye(P + 1), dense.Sigma, atol=1e-12)
    assert op.scalar_noise == dense.scalar_noise


@pytest.mark.parametrize('old, new', [('glm_problem', 'glmnet_problem'),
                                      ('glm_inference', 'glmnet_inference')])
def test_deprecated_module_names(old, new):
    # the modules' old names still work, with a DeprecationWarning
    import importlib, sys
    sys.modules.pop(f'glmnet.inference.{old}', None)
    with pytest.warns(DeprecationWarning, match=new):
        module = importlib.import_module(f'glmnet.inference.{old}')
    renamed = importlib.import_module(f'glmnet.inference.{new}')
    for name in renamed.__all__ + ['np']:
        assert getattr(module, name) is getattr(renamed, name)


def test_deprecated_glmnet_problem():
    # glmnet_problem, renamed problem_from_solution, still works with a
    # DeprecationWarning; as a name, glmnet.inference.glmnet_problem is the module
    import importlib, types, warnings
    from glmnet.inference import problem_from_solution
    import glmnet.inference.glmnet_problem as module
    assert isinstance(module, types.ModuleType) and module.problem_from_solution is problem_from_solution
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((40, 3)), rng.standard_normal(40)
    args = (X, y, np.array([0.5, 0., 0.]), 0.1, 0.2)
    expected = problem_from_solution(*args)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        old_module = importlib.import_module('glmnet.inference.glm_problem')
    from glmnet.inference import glmnet_problem
    for old in [glmnet_problem, old_module.glmnet_problem]:
        with pytest.warns(DeprecationWarning, match='problem_from_solution'):
            problem = old(*args)
        np.testing.assert_allclose(problem.G_hat, expected.G_hat)
        np.testing.assert_allclose(problem.beta_hat, expected.beta_hat)
