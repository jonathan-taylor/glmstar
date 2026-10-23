"""
Smoothers of one's own, the documentation's examples ("Writing a
smoother", docs/gam.md): a `Smoother` (a step function, the means in bins)
against `categorical()` of the bins; a `PenalizedSmoother` (a Fourier basis
with its roughness penalty) against `penalized_columns()` of its basis, by
backfitting and jointly, in a GAM and a GLMM. And the former names of the
smoothers, deprecated.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.gam import (GAM, GLMM, GAMControl, ExperimentalWarning, Smoother, PenalizedSmoother,
                        SplineSmoother, categorical, linear, penalized_columns, s)

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@dataclass
class BinSmoother(Smoother):
    """A step function: the weighted mean of the residuals in each of `n_bins` bins."""
    n_bins: int = 10

    _label = 'bins'

    def setup(self, x):
        x = np.asarray(x, float)
        self.edges_ = np.quantile(x, np.linspace(0, 1, self.n_bins + 1))
        self.codes_ = self._codes(x)
        return self

    def _codes(self, x):
        return np.clip(np.searchsorted(self.edges_, x, side='right') - 1, 0, self.n_bins - 1)

    def _means(self, r, w):
        sw = np.bincount(self.codes_, w, self.n_bins)
        swr = np.bincount(self.codes_, w * r, self.n_bins)
        return np.divide(swr, sw, out=np.zeros(self.n_bins), where=sw > 0)

    def smooth(self, r):
        return self._means(r, self.w_)[self.codes_]

    def edf(self):
        return float(np.sum(np.bincount(self.codes_, self.w_, self.n_bins) > 0) - 1)

    def finalize(self, f, r, coef=None):
        self.means_ = self._means(f, np.ones_like(f))
        return self

    def predict(self, x):
        return self.means_[self._codes(np.asarray(x, float))]


@dataclass
class FourierSmoother(PenalizedSmoother):
    """A periodic smooth: `K` sines and cosines, penalized by the integral of f''^2 over a period."""
    K: int = 5
    period: float = 1.

    _label = 'fourier'

    def basis(self, x=None):
        x = np.asarray(self.x_ if x is None else x, float)
        t = 2 * np.pi * np.outer(x, np.arange(1, self.K + 1)) / self.period
        return np.hstack([np.sin(t), np.cos(t)])

    def penalty_matrices(self):
        # the integral over a period of f''^2 for f = sum a_k sin + b_k cos
        d = (2 * np.pi * np.arange(1, self.K + 1) / self.period) ** 4 * self.period / 2
        return [np.diag(np.r_[d, d])]


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 500
    D = pd.DataFrame({'x': rng.uniform(0, 1, n), 'hour': rng.uniform(0, 24, n),
                      'z': rng.standard_normal(n)})
    D['y'] = (np.sin(2 * np.pi * D['hour'] / 24) + 0.5 * np.cos(4 * np.pi * D['hour'] / 24) +
              np.where(D['x'] > 0.5, 0.4, 0.) + 0.3 * D['z'] + 0.3 * rng.standard_normal(n))
    return D


def test_bin_smoother(data):
    G = GAM([BinSmoother('x', n_bins=8), linear('z')], control=GAMControl(**TIGHT)).fit(data, data['y'])
    term = G.terms_[0]
    D = data.assign(bin=term.codes_)
    H = GAM([categorical('bin'), linear('z')], control=GAMControl(**TIGHT)).fit(D, D['y'])
    np.testing.assert_allclose(G.predict(data), H.predict(D), rtol=1e-8)
    np.testing.assert_allclose(G.edf_.iloc[0], 7)
    assert G.terms_[0].name == 'bins(x)'
    # new values
    new = pd.DataFrame({'x': [0.1, 0.9], 'z': [0., 0.]})
    np.testing.assert_allclose(G.predict(new), H.predict(new.assign(bin=term._codes(new['x']))))
    # no basis: no joint solve, no covariance
    assert G.posterior_covariance_ is None


def _columns_equivalent(term, data, **kwargs):
    """`penalized_columns` of the term's basis and penalty."""
    F = term.setup(data['hour'].to_numpy())
    B = F.basis()
    cols = [f'b{j}' for j in range(B.shape[1])]
    D = pd.concat([data, pd.DataFrame(B, columns=cols, index=data.index)], axis=1)
    return D, penalized_columns(cols, F.penalty_matrices(), **kwargs)


@pytest.mark.parametrize('solver', ['backfit', 'joint'])
def test_fourier_fixed(data, solver):
    fourier = FourierSmoother('hour', K=4, period=24, lam=1e-4)
    G = GAM([fourier, linear('z')], control=GAMControl(solver=solver, **TIGHT)).fit(data, data['y'])
    D, cols = _columns_equivalent(FourierSmoother('hour', K=4, period=24), data, lam=[1e-4])
    H = GAM([cols, linear('z')], control=GAMControl(solver=solver, **TIGHT)).fit(D, D['y'])
    np.testing.assert_allclose(G.predict(data), H.predict(D), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(G.edf_.iloc[0], H.edf_.iloc[0], rtol=1e-8)
    # periodic: the same prediction a period later
    new = pd.DataFrame({'hour': [3., 27.], 'z': [0., 0.]})
    pred = G.predict(new)
    np.testing.assert_allclose(pred[0], pred[1], rtol=1e-10)
    # the posterior covariance, from its basis and penalty
    _, std = G.predict(new, return_std=True)
    assert np.all(std > 0)


def test_fourier_glmm(data):
    pytest.importorskip('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        G = GLMM([FourierSmoother('hour', K=6, period=24), linear('z')],
                 control=GAMControl(**TIGHT)).fit(data, data['y'])
        D, cols = _columns_equivalent(FourierSmoother('hour', K=6, period=24), data)
        H = GLMM([cols, linear('z')], control=GAMControl(**TIGHT)).fit(D, D['y'])
    assert G.smoothing_.converged
    np.testing.assert_allclose(G.smoothing_.lam, H.smoothing_.lam, rtol=1e-6)
    np.testing.assert_allclose(G.reml_score_, H.reml_score_, rtol=1e-10)
    assert list(G.variance_components_.index) == ['fourier(hour)', 'scale']


def test_glmm_needs_basis(data):
    pytest.importorskip('jax')
    with pytest.raises(ValueError, match='basis'):
        GLMM([BinSmoother('x'), s('hour')]).fit(data, data['y'])


def test_former_names():
    import glmnet.gam as gam
    import glmnet.gam.terms as terms
    import glmnet.gam.tensor as tensor
    with pytest.warns(FutureWarning, match='SplineSmoother'):
        assert gam.SplineTerm is SplineSmoother
    with pytest.warns(FutureWarning, match='Smoother'):
        assert terms.Term is Smoother
    with pytest.warns(FutureWarning, match='TensorSplineSmoother'):
        assert tensor.TensorTerm is gam.TensorSplineSmoother
    with pytest.warns(FutureWarning, match='GaussianProcessSmoother'):
        assert gam.GPTerm is gam.GaussianProcessSmoother
    with pytest.raises(AttributeError):
        gam.NoSuchTerm
    with pytest.warns(FutureWarning, match='random_effect'):
        term = gam.re('g', lam=1.)
    assert type(term) is gam.RandomEffectSmoother and term.name == 'random_effect(g, lam=1.0)'
