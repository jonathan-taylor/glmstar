"""
Schall's PQL for binary crossed random effects, the method written on the
page "Crossed random effects at scale" (copied here): its trace
approximation against the exact traces of the blocks of the inverse, its
convergence, and its fixed point.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, re, linear, SmoothingMethod
from glmnet.gam.reml import SmoothingSelection
from glmnet.gam.selection import estimated_terms


# the method of the documentation's page "Crossed random effects at scale"
class Schall(SmoothingMethod):
    """Schall's PQL for random intercepts, with the off-diagonal-free trace approximation."""
    name = 'Schall'
    joint = False
    covariance_edf = False

    def __init__(self, tol=1e-6, maxit=100):
        self.tol, self.maxit = tol, maxit

    def select(self, gam, dispersion=None):
        terms, estimated = gam.terms_, estimated_terms(gam)
        weight = np.asarray(gam._weight, float)
        W = weight.sum()
        phi = 1. if dispersion is None else dispersion
        s2, state = np.ones(len(estimated)), None
        for iteration in range(self.maxit):
            for j, v in zip(estimated, s2):
                terms[j].lam = phi / v / W
            state = gam.inner_fit(state)            # warm started
            mu = np.asarray(state.mu, float)
            w = weight * mu * (1 - mu) / phi        # the working weights (logit link)
            new = np.empty_like(s2)
            for i, j in enumerate(estimated):
                term = terms[j]
                R = len(term.levels_)
                W_g = np.bincount(term.codes_, weights=w, minlength=R)
                nu = np.sum(1 / (W_g + 1 / s2[i])) / s2[i]
                b = np.asarray(state.coefs[j])
                new[i] = b @ b / (R - nu)
            change = np.max(np.abs(new - s2) / s2)
            s2 = new
            if change < self.tol:
                break
        lam_raw = phi / s2
        for j, lam in zip(estimated, lam_raw):
            terms[j].lam = float(lam) / W
        return SmoothingSelection(method=self.name, terms=[terms[j].name for j in estimated],
                                  lam=lam_raw / W, lam_raw=lam_raw, theta=np.log(lam_raw),
                                  dispersion=phi, score=np.nan, gradient=np.zeros(0),
                                  converged=bool(change < self.tol),
                                  n_evaluations=iteration + 1)


def binary_ratings(N, R, C, seed=0):
    rng = np.random.default_rng(seed)
    customer = rng.integers(0, R, N)
    item = (rng.zipf(1.4, N) - 1) % C
    x = rng.standard_normal(R)[customer] + 0.5 * rng.standard_normal(N)
    eta = -0.5 + 0.3 * x + rng.normal(0, 1.0, R)[customer] + rng.normal(0, 0.5, C)[item]
    return pd.DataFrame({'customer': customer, 'item': item, 'x': x,
                         'y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})


@pytest.fixture(scope='module')
def fit():
    D = binary_ratings(4000, 300, 80)
    G = GAM(terms=[linear('x'), re('customer'), re('item')], family=sm.families.Binomial(),
            method=Schall(tol=1e-10), control=GAMControl(epsnr=1e-12, bf_tol=1e-13,
                                                         bf_maxit=5000)).fit(D, D['y'])
    return G, D


def test_converges_to_the_fixed_point(fit):
    # at convergence the variances satisfy Schall's equations at the final fit
    G, D = fit
    assert G.smoothing_.converged
    s2 = G.variance_components_['variance'].iloc[:2].values
    mu = np.asarray(G.state_.mu)
    w = mu * (1 - mu)
    for term, v in zip(G.terms_[1:], s2):
        W_g = np.bincount(term.codes_, weights=w, minlength=len(term.levels_))
        nu = np.sum(1 / (W_g + 1 / v)) / v
        b = term.effects_.to_numpy()
        np.testing.assert_allclose(b @ b / (len(term.levels_) - nu), v, rtol=1e-6)


def test_trace_approximation(fit):
    # the blocks' traces of T^{-1} without T's off-diagonal blocks (the
    # crossing), against the exact ones: close (Ghosh, Hastie & Owen)
    G, D = fit
    s2 = G.variance_components_['variance'].iloc[:2].values
    mu = np.asarray(G.state_.mu)
    w = mu * (1 - mu)
    Z = [G.terms_[j].basis() for j in (1, 2)]
    ZZ = np.hstack(Z)
    T = ZZ.T @ (w[:, None] * ZZ) + np.diag(np.r_[np.full(Z[0].shape[1], 1 / s2[0]),
                                                  np.full(Z[1].shape[1], 1 / s2[1])])
    Tinv = np.linalg.inv(T)
    R = Z[0].shape[1]
    exact = [np.trace(Tinv[:R, :R]), np.trace(Tinv[R:, R:])]
    approx = [np.sum(1 / (Zk.T @ w + 1 / v)) for Zk, v in zip(Z, s2)]
    np.testing.assert_allclose(approx, exact, rtol=0.05)


def test_against_laplace(fit):
    # PQL's variances are smaller than the Laplace approximate ML's (Breslow
    # & Clayton); the fixed effects are close
    pytest.importorskip('jax')
    G, D = fit
    L = GAM(terms=[linear('x'), re('customer'), re('item')], family=sm.families.Binomial(),
            method='ML').fit(D, D['y'])
    v, vL = G.variance_components_['variance'].iloc[:2], L.variance_components_['variance'].iloc[:2]
    assert np.all(v.values < vL.values)
    np.testing.assert_allclose(G.fixed_effects_['estimate'].values,
                               L.fixed_effects_['estimate'].values, atol=0.05)
