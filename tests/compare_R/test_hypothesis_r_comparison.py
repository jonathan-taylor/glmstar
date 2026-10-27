"""
`glmnet.hypothesis.anova_table` (type I) against R: `anova(lm(...))`,
`anova(glm(...), test="Chisq")` for a Poisson GLM with a factor and an interaction, and
`anova(coxph(...))` (sequential likelihood ratio tests).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from modelmatrix import ModelSpec as MS

from glmnet.glm import GLM, GLMFamilySpec, GLMControl
from glmnet.cox import CoxLM, CoxFamily
from glmnet.hypothesis import anova_table

TIGHT = GLMControl(epsnr=1e-12, mxitnr=100)


def _data(seed=3, n=300):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({'x': rng.standard_normal(n),
                      'c': pd.Categorical(rng.choice(list('abc'), n))})
    D['cnt'] = rng.poisson(np.exp(0.3 * D['x'] + 0.3 * (D['c'] == 'b')))
    D['time'] = rng.exponential(np.exp(-0.5 * D['x'] - 0.3 * (D['c'] == 'c'))) + 0.01
    D['status'] = rng.binomial(1, 0.8, n).astype(float)
    return D


def _R(Rinfo, D):
    rpy = Rinfo['rpy']
    rpy.globalenv['x'] = Rinfo['FloatVector'](D['x'].to_numpy())
    rpy.globalenv['cnt'] = Rinfo['FloatVector'](D['cnt'].to_numpy(float))
    rpy.globalenv['time'] = Rinfo['FloatVector'](D['time'].to_numpy())
    rpy.globalenv['status'] = Rinfo['FloatVector'](D['status'].to_numpy())
    rpy.r.assign('cc', rpy.StrVector(D['c'].astype(str).tolist()))
    rpy.r('cc <- factor(cc)')
    return rpy


def test_gaussian_type_I(Rinfo):
    D = _data()
    D['y'] = D['x'] + 0.4 * (D['c'] == 'b') + np.random.default_rng(0).standard_normal(len(D))
    rpy = _R(Rinfo, D)
    rpy.globalenv['y'] = Rinfo['FloatVector'](D['y'].to_numpy())
    table = anova_table(GLM(control=TIGHT), MS(['x', 'c', ('x', 'c')]), D, D['y'], type='I')
    rpy.r('a <- anova(lm(y ~ x + cc + x:cc))')
    for ours, R in [('df', 'Df'), ('deviance', 'Sum Sq'), ('F', 'F value'), ('P(>F)', 'Pr(>F)')]:
        np.testing.assert_allclose(table[ours].to_numpy()[:3],
                                   np.asarray(rpy.r(f'a[["{R}"]]'), float)[:3], rtol=1e-7)
    np.testing.assert_allclose(table.loc['Residuals', 'deviance'],
                               np.asarray(rpy.r('a[["Sum Sq"]]'), float)[3], rtol=1e-10)


def test_poisson_type_I(Rinfo):
    D = _data()
    rpy = _R(Rinfo, D)
    table = anova_table(GLM(family=GLMFamilySpec(sm.families.Poisson()), control=TIGHT),
                        MS(['x', 'c', ('x', 'c')]), D, D['cnt'], type='I')
    rpy.r('a <- anova(glm(cnt ~ x + cc + x:cc, family=poisson, '
          'control=glm.control(epsilon=1e-12)), test="Chisq")')
    dev = np.asarray(rpy.r('a$Deviance'), float)[1:]
    pval = np.asarray(rpy.r('a[["Pr(>Chi)"]]'), float)[1:]
    np.testing.assert_allclose(table['deviance'].iloc[:3], dev, rtol=1e-7)
    np.testing.assert_allclose(table['P(>chi2)'].iloc[:3], pval, rtol=1e-5)
    np.testing.assert_allclose(table.loc['Residuals', 'deviance'],
                               np.asarray(rpy.r('a[["Resid. Dev"]]'), float)[-1], rtol=1e-8)


def test_cox_type_I(Rinfo):
    D = _data()
    rpy = _R(Rinfo, D)
    table = anova_table(CoxLM(family=CoxFamily(event_id='time', status_id='status')),
                        MS(['x', 'c']), D, D, type='I')
    rpy.r('library(survival); a <- anova(coxph(Surv(time, status) ~ x + cc, '
          'ties="breslow", control=coxph.control(eps=1e-14, iter.max=100)))')
    chisq = np.asarray(rpy.r('a$Chisq'), float)[1:]
    pval = np.asarray(rpy.r('a[["Pr(>|Chi|)"]]'), float)[1:]
    np.testing.assert_allclose(table['chi2'].iloc[:2], chisq, rtol=1e-6)
    np.testing.assert_allclose(table['P(>chi2)'].iloc[:2], pval, rtol=1e-5)
