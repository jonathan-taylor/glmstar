"""
REML "by hand" (as docs/gam_reml.md): the Laplace approximate REML written
out in a few lines of JAX on the GAM's design and penalties, minimized
directly, against `GAM(method='REML')`; and, for the Gaussian, the
optimum's identities in the residuals (the dispersion, and each term's
prior size against its penalized degrees of freedom).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize

pytest.importorskip('scatter_smooth')
jax = pytest.importorskip('jax')
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy

from glmnet.gam import GAM, s


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 300
    D = pd.DataFrame({'x1': rng.uniform(0, 1, n), 'x2': rng.uniform(0, 1, n)})
    eta = np.sin(2 * np.pi * D['x1']) + 3 * (D['x2'] - 0.5)**2
    D['y'] = eta + 0.4 * rng.standard_normal(n)
    D['count'] = rng.poisson(np.exp(0.5 + 0.7 * eta))
    return D


def _by_hand(G, y, gaussian):
    """The REML optimum (log lam_raw, then log phi for the Gaussian) and criterion."""
    D = G.design_
    X, y = jnp.asarray(D.X), jnp.asarray(y, float)
    n, P = X.shape
    S = []
    for sl, S_j in zip(D.slices, D.penalties):
        E = np.zeros((P, P))
        E[sl, sl] = S_j
        S.append(jnp.asarray(E))
    ev = [np.linalg.eigvalsh(S_j) for S_j in D.penalties]
    ev = [e[e > e.max() * 1e-10] for e in ev]
    r, logdet_S1 = [len(e) for e in ev], [np.log(e).sum() for e in ev]
    M = P - sum(r)
    K = len(S)

    def S_lam(rho):
        return sum(jnp.exp(rho_j) * S_j for rho_j, S_j in zip(rho, S))

    if gaussian:
        deviance = lambda beta: jnp.sum((y - X @ beta)**2)
        hessian = lambda beta, rho: X.T @ X + S_lam(rho)
        ls = lambda phi: -n / 2 * jnp.log(2 * np.pi * phi)
        start = y.mean()
    else:
        deviance = lambda beta: 2 * jnp.sum(xlogy(y, y) - xlogy(y, jnp.exp(X @ beta)) -
                                            (y - jnp.exp(X @ beta)))
        hessian = lambda beta, rho: X.T @ (jnp.exp(X @ beta)[:, None] * X) + S_lam(rho)
        ls = lambda phi: jnp.sum(xlogy(y, y) - y - gammaln(y + 1))
        start = jnp.log(y.mean())

    def beta_hat(rho):
        # Newton's method on the penalized deviance over 2
        beta = jnp.zeros(P).at[0].set(start)
        grad = jax.grad(lambda b: deviance(b) / 2 + b @ S_lam(rho) @ b / 2)
        for _ in range(30):
            beta = beta - jnp.linalg.solve(hessian(beta, rho), grad(beta))
        return beta

    def V(params):
        rho = params[:K]
        phi = jnp.exp(params[K]) if gaussian else 1.
        beta = beta_hat(rho)
        Dp = deviance(beta) + beta @ S_lam(rho) @ beta
        logdet_S = sum(r_j * rho_j + l_j for r_j, rho_j, l_j in zip(r, rho, logdet_S1))
        return (Dp / (2 * phi) - ls(phi) + jnp.linalg.slogdet(hessian(beta, rho))[1] / 2 -
                logdet_S / 2 - M * jnp.log(2 * np.pi * phi) / 2)

    value_and_grad = jax.jit(jax.value_and_grad(V))
    fit = minimize(lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
                   np.zeros(K + gaussian), jac=True, method='BFGS', options=dict(gtol=1e-9))
    return fit.x, fit.fun, M, r


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_reml_by_hand(data, family):
    gaussian = family == 'gaussian'
    response = 'y' if gaussian else 'count'
    fam = sm.families.Gaussian() if gaussian else sm.families.Poisson()
    G = GAM(terms=[s('x1'), s('x2')], family=fam, method='REML').fit(data, data[response])
    sel = G.smoothing_
    with jax.enable_x64(True):
        x, value, _, _ = _by_hand(G, data[response].values, gaussian)
    np.testing.assert_allclose(x[:2], np.log(sel.lam_raw), atol=1e-5)
    if gaussian:
        np.testing.assert_allclose(x[2], np.log(sel.dispersion), atol=1e-6)
    np.testing.assert_allclose(value, sel.score, rtol=1e-9)


def test_gaussian_identities(data):
    # at the REML optimum: phi = (RSS + b' S b) / (n - M), and
    # lam_j b_j' S_j b_j / phi = r_j - lam_j tr(H^{-1} S_j) = edf_j - (K_j - r_j)
    G = GAM(terms=[s('x1'), s('x2')], method='REML').fit(data, data['y'])
    D, sel = G.design_, G.smoothing_
    y, n = data['y'].values, data.shape[0]
    lam, phi = sel.lam_raw, sel.dispersion
    S_hat = D.penalty(lam)
    H = D.X.T @ D.X + S_hat
    beta = np.linalg.solve(H, D.X.T @ y)
    np.testing.assert_allclose(D.X @ beta, G.predict(data), rtol=1e-8)
    resid = y - D.X @ beta
    r = [np.linalg.matrix_rank(S_j) for S_j in D.penalties]
    M = D.n_coef - sum(r)
    np.testing.assert_allclose((resid @ resid + beta @ S_hat @ beta) / (n - M), phi, rtol=1e-8)
    Hinv = np.linalg.inv(H)
    for j, (sl, S_j) in enumerate(zip(D.slices, D.penalties)):
        size = lam[j] * beta[sl] @ S_j @ beta[sl] / phi
        np.testing.assert_allclose(size, r[j] - lam[j] * np.trace(Hinv[sl, sl] @ S_j), rtol=1e-6)
        np.testing.assert_allclose(size, G.edf_.iloc[j] - (S_j.shape[0] - r[j]), rtol=1e-6)
