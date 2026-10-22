"""
Variance components by the method of moments (`CrossedMoments`) and
custom `SmoothingMethod`s: the U statistics' expectations exactly (as traces
of quadratic forms), the balanced crossed design's ANOVA estimators, nested
effects, a large sparse crossed design, and a user's method.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, re, linear, s, CrossedMoments, SmoothingMethod
from glmnet.gam.moments import moment_equations
from glmnet.gam.selection import estimated_terms


def _indicators(c):
    return (c[:, None] == np.arange(c.max() + 1)[None, :]).astype(float)


def test_moment_equations_are_unbiased():
    # U = r' Q r, so E[U] = tr(Q V), V = sum_m s_m^2 Z_m Z_m' + s_E^2 I: the
    # coefficients are tr(Q Z_m Z_m') and tr(Q), on an unbalanced design
    rng = np.random.default_rng(0)
    N = 60
    codes = [rng.integers(0, 5, N), rng.integers(0, 7, N), rng.integers(0, 3, N)]
    _, A = moment_equations(rng.standard_normal(N), codes)
    Qs = []
    for c in codes:
        Z = _indicators(c)
        P = Z @ np.diag(1 / np.maximum(Z.sum(0), 1)) @ Z.T     # the levels' means
        Qs.append(np.identity(N) - P)
    Qs.append(N * (np.identity(N) - np.ones((N, N)) / N))
    for i, Q in enumerate(Qs):
        expected = [np.trace(Q @ _indicators(c) @ _indicators(c).T) for c in codes] + [np.trace(Q)]
        np.testing.assert_allclose(A[i], expected, rtol=1e-12, atol=1e-9)


def test_balanced_is_anova():
    rng = np.random.default_rng(0)
    R, C = 30, 20
    row, col = np.repeat(np.arange(R), C), np.tile(np.arange(C), R)
    y = 2 + rng.normal(0, 1.5, R)[row] + rng.normal(0, 0.8, C)[col] + rng.normal(0, 1, R * C)
    D = pd.DataFrame({'row': row, 'col': col, 'y': y})
    G = GAM(terms=[re('row'), re('col')], method=CrossedMoments()).fit(D, D['y'])
    Y = y.reshape(R, C)
    m = Y.mean()
    MSA = C * ((Y.mean(1) - m)**2).sum() / (R - 1)
    MSB = R * ((Y.mean(0) - m)**2).sum() / (C - 1)
    MSE = ((Y - Y.mean(1)[:, None] - Y.mean(0)[None] + m)**2).sum() / ((R - 1) * (C - 1))
    np.testing.assert_allclose(G.variance_components_['variance'].values,
                               [(MSA - MSE) / C, (MSB - MSE) / R, MSE], rtol=1e-10)
    assert G.smoothing_.method == 'moments'
    np.testing.assert_allclose(G.dispersion_, MSE, rtol=1e-10)


def test_large_crossed():
    # 200,000 observations of 2000 x 500 crossed levels, the columns' sizes
    # very unequal, a fixed slope: in linear time, near the variances
    rng = np.random.default_rng(1)
    N, R, C = 200000, 2000, 500
    row = rng.integers(0, R, N)
    col = (rng.zipf(1.5, N) - 1) % C
    a, b = rng.normal(0, 1.0, R), rng.normal(0, 0.5, C)
    x = rng.standard_normal(N)
    y = 1 + 0.5 * x + a[row] + b[col] + rng.normal(0, 2, N)
    D = pd.DataFrame({'row': row, 'col': col, 'x': x, 'y': y})
    G = GAM(terms=[linear('x'), re('row'), re('col')], method=CrossedMoments()).fit(D, D['y'])
    v = G.variance_components_['variance']
    # (against the effects' realized variances, the columns' that occur)
    np.testing.assert_allclose(v['re(row)'], a.var(), rtol=0.1)
    np.testing.assert_allclose(v['re(col)'], b[np.unique(col)].var(), rtol=0.5)
    np.testing.assert_allclose(v['scale'], 4, rtol=0.02)
    np.testing.assert_allclose(G.terms_[0].coef_, 0.5, atol=0.02)
    # the predicted effects are the backfitted BLUPs at these variances
    assert np.corrcoef(G.terms_[1].effects_.values, a)[0, 1] > 0.95


def test_nested_and_fixed_smooth():
    rng = np.random.default_rng(2)
    n = 3000
    batch = rng.integers(0, 30, n)
    cask = rng.integers(0, 4, n)
    x = rng.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x) + rng.normal(0, 1, 30)[batch] +
         rng.normal(0, 0.6, (30, 4))[batch, cask] + rng.normal(0, 0.5, n))
    D = pd.DataFrame({'batch': batch, 'cask': cask, 'x': x, 'y': y})
    G = GAM(terms=[s('x', df=5), re('batch'), re(['batch', 'cask'])],
            method=CrossedMoments()).fit(D, D['y'])
    v = G.variance_components_['variance']
    # (as gam.vcomp, also the spline's)
    assert {'re(batch)', 're(batch:cask)', 'scale'} <= set(v.index)
    np.testing.assert_allclose(v['scale'], 0.25, rtol=0.1)
    assert np.all(v[['re(batch)', 're(batch:cask)']] > 0)
    assert list(G.smoothing_.terms) == ['re(batch)', 're(batch:cask)']


def test_custom_method():
    # a user's method: the variance of re(g) fixed at a given value
    class GivenVariance(SmoothingMethod):
        name = 'given'
        experimental = False

        def __init__(self, variance):
            self.variance = variance

        def select(self, gam, dispersion=None):
            from glmnet.gam.reml import SmoothingSelection
            (j,) = estimated_terms(gam)
            W = gam._weight.sum()
            lam = dispersion / self.variance / W
            gam.terms_[j].lam = lam
            return SmoothingSelection(method=self.name, terms=[gam.terms_[j].name],
                                      lam=np.array([lam]), lam_raw=np.array([lam * W]),
                                      theta=np.log([lam * W]), dispersion=dispersion,
                                      score=np.nan, gradient=np.zeros(0), converged=True,
                                      n_evaluations=0)

    rng = np.random.default_rng(3)
    D = pd.DataFrame({'g': rng.integers(0, 10, 200), 'x': rng.uniform(size=200)})
    D['y'] = D['x'] + rng.normal(0, 1, 10)[D['g']] + rng.normal(0, 1, 200)
    G = GAM(terms=[linear('x'), re('g')], method=GivenVariance(2.)).fit(D, D['y'], dispersion=1.)
    G0 = GAM(terms=[linear('x'), re('g', lam=0.5 / 200)]).fit(D, D['y'], dispersion=1.)
    np.testing.assert_allclose(G.predict(D), G0.predict(D), rtol=1e-10)
    np.testing.assert_allclose(G.variance_components_.loc['re(g)', 'variance'], 2.)
    assert G.smoothing_.method == 'given'


def test_errors():
    rng = np.random.default_rng(4)
    D = pd.DataFrame({'g': rng.integers(0, 10, 200), 'x': rng.uniform(size=200),
                      'w': rng.uniform(1, 2, 200)})
    D['y'] = rng.normal(size=200)
    D['yb'] = rng.integers(0, 2, 200)
    with pytest.raises(ValueError, match='Gaussian'):
        GAM(terms=[re('g')], family=sm.families.Binomial(),
            method=CrossedMoments()).fit(D, D['yb'])
    with pytest.raises(ValueError, match='equal prior weights'):
        GAM(terms=[re('g')], method=CrossedMoments(), weight_id='w',
            response_id='y').fit(D, D[['y', 'w']])
    with pytest.raises(ValueError, match='random intercepts'):
        GAM(terms=[re('g', by='x')], method=CrossedMoments()).fit(D, D['y'])
    with pytest.raises(ValueError, match="method should be one of"):
        GAM(terms=[re('g')], method='GCV').fit(D, D['y'])


@pytest.fixture(scope='module')
def crossed():
    from glmnet.gam.terms import ColumnsTerm  # noqa: F401
    rng = np.random.default_rng(0)
    N, R, C = 3000, 60, 40
    row, col = rng.integers(0, R, N), rng.integers(0, C, N)
    D = pd.DataFrame({'row': row, 'col': col, 'x': rng.standard_normal(N),
                      'z1': rng.standard_normal(N), 'z2': rng.uniform(size=N)})
    D['y'] = (1 + 0.5 * D['x'] + 0.2 * D['z1'] - D['z2'] + rng.normal(0, 1, R)[row] +
              rng.normal(0, 0.6, C)[col] + rng.normal(0, 1, N))
    return D


@pytest.mark.parametrize('method', ['REML', 'moments'])
def test_fixed_effects_by_backfitting(crossed, method):
    # the backfitted covariance of the linear coefficients is the block of
    # the posterior covariance (the dense design)
    from glmnet.gam.terms import ColumnsTerm
    D = crossed
    if method == 'REML':
        pytest.importorskip('jax')
    G = GAM(terms=[linear('x'), ColumnsTerm(['z1', 'z2']), re('row'), re('col')],
            method=CrossedMoments() if method == 'moments' else method,
            control=GAMControl(bf_tol=1e-13, bf_maxit=5000, epsnr=1e-12)).fit(D, D['y'])
    fe = G.fixed_effects_
    design = G.design_
    idx = [0, design.slices[0].start, design.slices[1].start, design.slices[1].start + 1]
    np.testing.assert_allclose(G.fixed_effects_covariance_.values,
                               G.posterior_covariance_[np.ix_(idx, idx)], rtol=1e-8)
    np.testing.assert_allclose(fe['estimate'].values[1:], [0.5, 0.2, -1], atol=0.1)
    assert list(fe.index)[:2] == ['intercept', 'linear(x)']


def test_fixed_effects_lmer(Rinfo, crossed):
    # REML: lme4's fixed effects and vcov (the GLS standard errors)
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("lme4", quietly=TRUE))')[0]:
        pytest.skip('requires the R package lme4')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', crossed)
    rpy.r("""m <- lme4::lmer(y ~ x + z1 + z2 + (1 | row) + (1 | col), data=D, REML=TRUE,
                         control=lme4::lmerControl(optimizer='bobyqa', optCtrl=list(rhoend=1e-12)))""")
    G = GAM(terms=[linear('x'), linear('z1'), linear('z2'), re('row'), re('col')], method='REML',
            control=GAMControl(bf_tol=1e-13, bf_maxit=5000, epsnr=1e-12)).fit(crossed, crossed['y'])
    with Rinfo['np_cv_rules'].context():
        se_R = np.sqrt(np.diag(np.asarray(rpy.r('as.matrix(vcov(m))'))))
        beta_R = np.asarray(rpy.r('lme4::fixef(m)'))
    fe = G.fixed_effects_
    # (the slopes; lme4's intercept is at x = z = 0, ours at their means)
    np.testing.assert_allclose(fe['estimate'].values[1:], beta_R[1:], rtol=1e-5)
    np.testing.assert_allclose(fe['std_error'].values[1:], se_R[1:], rtol=1e-4)


def test_fixed_effects_large():
    # 200,000 observations of 2000 x 500 crossed levels: the standard errors
    # by backfitting, without the design (2501 columns)
    rng = np.random.default_rng(1)
    N, R, C = 200000, 2000, 500
    row, col = rng.integers(0, R, N), (rng.zipf(1.5, N) - 1) % C
    x = rng.standard_normal(N)
    y = 1 + 0.5 * x + rng.normal(0, 1, R)[row] + rng.normal(0, 0.5, C)[col] + rng.normal(0, 2, N)
    D = pd.DataFrame({'row': row, 'col': col, 'x': x, 'y': y})
    G = GAM(terms=[linear('x'), re('row'), re('col')], method=CrossedMoments()).fit(D, D['y'])
    fe = G.fixed_effects_
    assert G._covariance is None          # (the dense covariance was not formed)
    # near the OLS standard error of the slope: x is independent of the effects
    se_ols = np.sqrt(G.dispersion_ / np.sum((x - x.mean())**2))
    np.testing.assert_allclose(fe.loc['linear(x)', 'std_error'], se_ols, rtol=0.05)
    assert 'edf_influence' not in G.summary()
