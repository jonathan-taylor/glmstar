import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import statsmodels.genmod.families as smf

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl

from glmnet.inference import glmnet_problem, glmstar_problem
from glmnet.inference import glm_inference_problem, glmstar_inference_problem

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
    # glm_inference from a GLMProblem, with weights, offset and a ridge term
    X, df, rows = data
    y, w, o = df['poisson'].values, df['w'].values, df['o'].values
    lam, alpha = 0.05, 0.5
    # any coefficients will do for the algebra
    coef = np.r_[0.25, -0.2, 0.1, np.zeros(P - 3)]
    prob = glmnet_problem(X[rows], y[rows], coef, 0.1, lam, family='poisson', weights=w[rows],
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
    np.testing.assert_allclose(li.Sigma, prob.Q_hat / w.sum())


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
