"""
`glmnet.irls.IRLS`: the convergence criteria, the regularizer hook
`has_objective`, and the iteration count `n_iter_`.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pytest
import statsmodels.api as sm

from glmnet import GLM
from glmnet.glm import GLMControl, GLMRegularizer


def _logistic_data(n=200, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X @ np.array([1., -0.5, 0.25, 0.]) + 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return X, y


@dataclass
class _OvershootRegularizer(GLMRegularizer):
    """Three times the Newton step: the objective increases, so steps are halved."""

    def newton_step(self, design, pseudo_response, sample_weight, cur_state):
        new = super().newton_step(design, pseudo_response, sample_weight, cur_state)
        klass = cur_state.__class__
        return klass(3 * new.coef - 2 * cur_state.coef,
                     3 * new.intercept - 2 * cur_state.intercept)


@dataclass
class _NoObjectiveRegularizer(GLMRegularizer):
    """A regularizer whose steps are not checked against the objective."""
    has_objective: bool = False


def _fit(X, y, regularizer=None, **control):
    G = GLM(family=sm.families.Binomial(), control=GLMControl(**control))
    if regularizer is not None:
        regularizer = regularizer(fit_intercept=True)
    return G.fit(X, y, regularizer=regularizer)


def test_n_iter():
    X, y = _logistic_data()
    G = _fit(X, y)
    assert 1 < G.n_iter_ < GLMControl.mxitnr

    # the Gaussian with identity link: one step
    rng = np.random.default_rng(1)
    G = GLM().fit(X, X @ np.ones(X.shape[1]) + rng.standard_normal(X.shape[0]))
    assert G.n_iter_ == 1


def test_eta_convergence():
    X, y = _logistic_data()
    G_obj = _fit(X, y)
    G_eta = _fit(X, y, convergence='eta')
    np.testing.assert_allclose(G_eta.coef_, G_obj.coef_, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(G_eta.intercept_, G_obj.intercept_, rtol=1e-7, atol=1e-9)

    # statsmodels
    res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(G_eta.coef_, res.params[1:], rtol=1e-6, atol=1e-8)


def test_bad_convergence():
    X, y = _logistic_data()
    with pytest.raises(ValueError, match='convergence'):
        _fit(X, y, convergence='coef')


def test_step_halving():
    # undamped, the step diverges; it is halved, to 1.5 times the Newton step,
    # when the objective increases. Near the minimum, increases below the
    # tolerance of the descent check (1e-7) are not halved, so the iterations
    # can cycle there: only the neighbourhood of the minimum is checked
    X, y = _logistic_data()
    G = _fit(X, y)
    with warnings.catch_warnings():
        # the cycling fit may not converge
        warnings.simplefilter('ignore', UserWarning)
        G_over = _fit(X, y, regularizer=_OvershootRegularizer, mxitnr=50)
    np.testing.assert_allclose(G_over.coef_, G.coef_, rtol=1e-3, atol=1e-3)


def test_no_objective():
    # without an objective the steps are not halved for it, and the iterations
    # stop on the change in the linear predictor
    X, y = _logistic_data()
    G = _fit(X, y)
    G_none = _fit(X, y, regularizer=_NoObjectiveRegularizer)
    np.testing.assert_allclose(G_none.coef_, G.coef_, rtol=1e-7, atol=1e-9)
    assert G_none.n_iter_ <= GLMControl.mxitnr
