"""
`glmnet.hypothesis`: general linear hypotheses, ANOVA tables (types I and
III) of a modelmatrix ModelSpec's terms, and the sandwich with a meat of
one's own, against statsmodels.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from modelmatrix import ModelSpec as MS, contrast

from glmnet.glm import GLM, GLMFamilySpec, GLMControl, summarize, leverage
from glmnet.cox import CoxLM, CoxFamily
from glmnet.hypothesis import (linear_hypothesis, anova_table, sandwich_covariance,
                               score_contributions)

TIGHT = GLMControl(epsnr=1e-12, mxitnr=100)
POISSON = GLMFamilySpec(sm.families.Poisson())


def _data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({'x': rng.standard_normal(n),
                      'z': rng.standard_normal(n),
                      'c': pd.Categorical(rng.choice(list('abc'), n)),
                      'g': rng.integers(0, 25, n),
                      'w': rng.uniform(0.5, 2, n)})
    D['y'] = 1 + D['x'] + 0.5 * (D['c'] == 'b') + 0.3 * D['x'] * (D['c'] == 'c') \
        + rng.standard_normal(n)
    D['cnt'] = rng.poisson(np.exp(0.3 * D['x'] + 0.2 * (D['c'] == 'b')))
    D['time'] = rng.exponential(np.exp(-0.5 * D['x'])) + 0.01
    D['status'] = rng.binomial(1, 0.8, n)
    return D


SPEC = MS(['x', 'c', ('x', 'c')])


def _fits(D, family='gaussian', **kw):
    X = SPEC.fit(D).transform(D)
    response = {'gaussian': 'y', 'poisson': 'cnt'}[family]
    fam = {'gaussian': {}, 'poisson': dict(family=POISSON)}[family]
    fit = GLM(summarize=True, control=TIGHT, **fam, **kw).fit(X, D[response])
    smfam = {'gaussian': sm.families.Gaussian(), 'poisson': sm.families.Poisson()}[family]
    return fit, X, sm.add_constant(X), D[response], smfam


def test_ols_matches_statsmodels():
    D = _data()
    fit, X, X1, y, _ = _fits(D)
    ols = sm.OLS(y, X1).fit()
    h = linear_hypothesis(fit, {'c[b]': 1, 'c[c]': -1})
    t = ols.t_test('c[b] - c[c] = 0')
    np.testing.assert_allclose(h.estimates['estimate'], np.ravel(t.effect), rtol=1e-8)
    np.testing.assert_allclose(h.estimates['std err'], np.ravel(t.sd), rtol=1e-8)
    np.testing.assert_allclose(h.estimates['P>|t|'], np.ravel(t.pvalue), rtol=1e-6)
    assert list(h.estimates.index) == ['c[b] - c[c]']
    # several rows: F, with a right-hand side
    h = linear_hypothesis(fit, [{'c[b]': 1}, {'x': 1}], rhs=[0, 1])
    f = ols.f_test('c[b] = 0, x = 1')
    assert h.test == 'F' and h.df == (2, fit.df_resid_)
    np.testing.assert_allclose(h.statistic, np.ravel(f.fvalue)[0], rtol=1e-8)
    np.testing.assert_allclose(h.p_value, float(f.pvalue), rtol=1e-6)
    # the intercept enters only if named: the mean of level b at x = 0
    h = linear_hypothesis(fit, {'intercept': 1, 'c[b]': 1})
    np.testing.assert_allclose(h.estimates['estimate'].iloc[0],
                               fit.params_['intercept'] + fit.params_['c[b]'])


def test_forms_agree():
    D = _data()
    fit = _fits(D)[0]
    names = list(fit.feature_names_)
    row = np.zeros(len(names))
    row[names.index('c[b]')], row[names.index('c[c]')] = 1, -1
    forms = [{'c[b]': 1, 'c[c]': -1},
             pd.Series({'c[b]': 1, 'c[c]': -1}),
             pd.DataFrame([{'c[b]': 1, 'c[c]': -1}], index=['b vs c']),
             row, row[None, :]]
    ref = linear_hypothesis(fit, forms[0])
    for L in forms[1:]:
        h = linear_hypothesis(fit, L)
        np.testing.assert_allclose(h.estimates.iloc[:, :6].to_numpy(),
                                   ref.estimates.iloc[:, :6].to_numpy())
        assert h.statistic == pytest.approx(ref.statistic)


def test_errors():
    D = _data()
    fit = _fits(D)[0]
    p = len(fit.params_)
    with pytest.raises(ValueError, match='intercept first'):
        linear_hypothesis(fit, np.ones(p - 1))
    with pytest.raises(ValueError, match='unknown coefficients'):
        linear_hypothesis(fit, {'nope': 1})
    with pytest.raises(ValueError, match='linearly dependent'):
        linear_hypothesis(fit, [{'x': 1}, {'x': 2}])
    with pytest.raises(ValueError, match="'model', 'sandwich'"):
        linear_hypothesis(fit, {'x': 1}, covariance='HC3')


def test_poisson_wald_and_sandwich():
    D = _data()
    fit, X, X1, y, fam = _fits(D, 'poisson')
    ref = sm.GLM(y, X1, family=fam).fit(tol=1e-12)
    L = [{'c[b]': 1}, {'c[c]': 1}]
    h = linear_hypothesis(fit, L)
    w = ref.wald_test('c[b] = 0, c[c] = 0', use_f=False, scalar=True)
    assert h.test == 'chi2'
    np.testing.assert_allclose(h.statistic, w.statistic, rtol=1e-6)
    # HC0, from the fit or by name
    hc0 = sm.GLM(y, X1, family=fam).fit(tol=1e-12, cov_type='HC0')
    np.testing.assert_allclose(sandwich_covariance(fit).to_numpy(), hc0.cov_params().to_numpy(),
                               rtol=1e-6, atol=1e-12)
    h = linear_hypothesis(fit, L, covariance='sandwich')
    w = hc0.wald_test('c[b] = 0, c[c] = 0', use_f=False, scalar=True)
    np.testing.assert_allclose(h.statistic, w.statistic, rtol=1e-6)
    # sandwich=True's covariance is the sandwich's, HC0
    robust = GLM(family=POISSON, summarize=True, sandwich=True, control=TIGHT).fit(X, y)
    np.testing.assert_allclose(robust.covariance_, sandwich_covariance(fit).to_numpy(),
                               rtol=1e-10)


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_meat_and_groups(family):
    D = _data()
    fit, X, X1, y, fam = _fits(D, family)
    ref = sm.GLM(y, X1, family=fam)
    cluster = ref.fit(tol=1e-12, cov_type='cluster',
                      cov_kwds={'groups': D['g'], 'use_correction': False})
    C = sandwich_covariance(fit, groups=D['g'])
    np.testing.assert_allclose(C.to_numpy(), cluster.cov_params().to_numpy(), rtol=1e-6,
                               atol=1e-12)
    # a meat of one's own: HC3, from the scores and the leverage
    S = score_contributions(fit).to_numpy()
    h = np.asarray(leverage(fit, X))
    meat = (S / (1 - h)[:, None]).T @ (S / (1 - h)[:, None])
    if family == 'gaussian':
        hc3 = sm.OLS(y, X1).fit(cov_type='HC3')
        np.testing.assert_allclose(sandwich_covariance(fit, meat=meat).to_numpy(),
                                   hc3.cov_params().to_numpy(), rtol=1e-6)
    # in a summary, and a test
    table = summarize(fit, covariance=C)
    np.testing.assert_allclose(table['std err'], np.sqrt(np.diag(C)))
    assert 'z' in table.columns
    h = linear_hypothesis(fit, {'x': 1}, covariance=C)
    np.testing.assert_allclose(h.estimates['std err'].iloc[0], np.sqrt(C.loc['x', 'x']))
    with pytest.raises(ValueError, match='meat or groups'):
        sandwich_covariance(fit, meat=meat, groups=D['g'])


def test_type_I_matches_statsmodels():
    D = _data()
    table = anova_table(GLM(control=TIGHT), SPEC, D, D['y'], type='I')
    # statsmodels' anova_lm of the same sequence of columns
    X1 = sm.add_constant(SPEC.fit(D).transform(D))
    fits = [sm.OLS(D['y'], X1.iloc[:, :k]).fit() for k in (1, 2, 4, 6)]
    from statsmodels.stats.anova import anova_lm
    ref = anova_lm(*fits)
    np.testing.assert_allclose(table['deviance'].iloc[:3], ref['ss_diff'].iloc[1:], rtol=1e-8)
    np.testing.assert_allclose(table['df'].iloc[:3], ref['df_diff'].iloc[1:])
    np.testing.assert_allclose(table['F'].iloc[:3], ref['F'].iloc[1:], rtol=1e-8)
    # (R's p-values, on the largest model's residual df; anova_lm's are on each
    # model's own: tests/compare_R/test_hypothesis_r_comparison.py)
    from scipy.stats import f as f_dbn
    np.testing.assert_allclose(table['P(>F)'].iloc[:3],
                               f_dbn.sf(ref['F'].iloc[1:], ref['df_diff'].iloc[1:],
                                        fits[-1].df_resid), rtol=1e-6)
    assert list(table.index) == ['x', 'c', 'x:c', 'Residuals']
    assert table.loc['Residuals', 'df'] == fits[-1].df_resid


def test_type_III_matches_statsmodels():
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    D = _data()
    spec = MS(['x', contrast('c', 'sum'), ('x', contrast('c', 'sum'))])
    table = anova_table(GLM(control=TIGHT), spec, D, D['y'], type='III')
    ref = anova_lm(smf.ols('y ~ x + C(c, Sum) + x:C(c, Sum)', D).fit(), typ=3)
    for ours, theirs in [('intercept', 'Intercept'), ('x', 'x'), ('c', 'C(c, Sum)'),
                         ('x:c', 'x:C(c, Sum)')]:
        np.testing.assert_allclose(table.loc[ours, 'F'], ref.loc[theirs, 'F'], rtol=1e-8)
        np.testing.assert_allclose(table.loc[ours, 'P(>F)'], ref.loc[theirs, 'PR(>F)'],
                                   rtol=1e-6)
    # treatment contrasts in an interaction: a warning
    with pytest.warns(UserWarning, match='contrasts'):
        anova_table(GLM(), SPEC, D, D['y'], type='III')
    # a robust table
    C = anova_table(GLM(), spec, D, D['y'], type='III', covariance='sandwich')
    assert 'chi2' in C.columns


def test_cox_type_III_is_wald():
    D = _data()
    fam = CoxFamily(event_id='time', status_id='status')
    spec = MS(['x', 'c'])
    table = anova_table(CoxLM(family=fam), spec, D, D, type='III')
    X = spec.fit(D).transform(D)
    fit = CoxLM(family=fam, summarize=True).fit(X, D)
    h = linear_hypothesis(fit, [{'c[b]': 1}, {'c[c]': 1}])
    np.testing.assert_allclose(table.loc['c', 'chi2'], h.statistic)
    assert 'intercept' not in table.index


def test_spec_with_intercept_raises():
    D = _data()
    with pytest.raises(ValueError, match='no intercept'):
        anova_table(GLM(), MS(['x'], intercept=True), D, D['y'])
