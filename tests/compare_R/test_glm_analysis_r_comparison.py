"""
The analysis of unpenalized fits against R: `coxph` (coefficients, standard
errors, partial log-likelihood, AIC, martingale and deviance residuals, anova)
and `glm` (`hatvalues`, `anova(test="Chisq")`, and `test="F"` for the Gaussian).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet import GLM
from glmnet.glm import anova, leverage, residuals
from glmnet.cox import CoxLM, CoxFamilySpec
from glmnet.glm import GLMControl

TIGHT = GLMControl(epsnr=1e-12, mxitnr=100)


def _to_R(Rinfo, **arrays):
    rpy = Rinfo['rpy']
    for name, value in arrays.items():
        rpy.globalenv[name] = Rinfo['FloatVector'](np.asarray(value, float))


@pytest.mark.parametrize('ties', ['breslow', 'efron'])
@pytest.mark.parametrize('weighted', [False, True])
def test_coxph(Rinfo, ties, weighted):
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 3))
    t = rng.exponential(np.exp(-X @ [0.5, -0.5, 0]))
    c = rng.exponential(1.5, n)
    event = np.round(np.minimum(t, c), 1) + 0.05        # with ties
    status = (t <= c).astype(float)
    w = rng.uniform(0.5, 2, n) if weighted else np.ones(n)
    Y = pd.DataFrame({'event': event, 'status': status, 'w': w})
    family = CoxFamilySpec(event_data=Y, tie_breaking=ties)
    # CoxLM's Newton steps match R at the default control
    fit = CoxLM(family=family, summarize=True, weight_id='w').fit(X, Y)
    small = CoxLM(family=family, weight_id='w', exclude=[2]).fit(X, Y)
    robust = CoxLM(family=family, summarize=True, sandwich=True, weight_id='w').fit(X, Y)

    _to_R(Rinfo, x0=X[:, 0], x1=X[:, 1], x2=X[:, 2], event=event, status=status, w=w)
    rpy = Rinfo['rpy']
    rpy.r(f'''library(survival)
f <- coxph(Surv(event, status) ~ x0 + x1 + x2, weights=w, ties="{ties}", robust=FALSE,
           control=coxph.control(eps=1e-14, iter.max=100))
f1 <- coxph(Surv(event, status) ~ x0 + x1, weights=w, ties="{ties}", robust=FALSE,
           control=coxph.control(eps=1e-14, iter.max=100))
a <- anova(f1, f)
fr <- coxph(Surv(event, status) ~ x0 + x1 + x2, weights=w, ties="{ties}", robust=TRUE,
           control=coxph.control(eps=1e-14, iter.max=100))''')
    get = lambda expr: np.asarray(rpy.r(expr), float)

    np.testing.assert_allclose(fit.params_.values, get('coef(f)'), rtol=1e-6)
    np.testing.assert_allclose(fit.summary_['std err'].values, get('sqrt(diag(vcov(f)))'), rtol=1e-6)
    np.testing.assert_allclose(fit.loglike_, get('f$loglik[2]')[0], rtol=1e-10)
    np.testing.assert_allclose(fit.aic_, get('AIC(f)')[0], rtol=1e-10)
    np.testing.assert_allclose(residuals(fit), get('residuals(f, "martingale")'), atol=1e-5)
    np.testing.assert_allclose(residuals(fit, 'deviance'), get('residuals(f, "deviance")'), atol=1e-5)
    table = anova(small, fit)
    np.testing.assert_allclose(table['chi2'].iloc[1], get('a[2, "Chisq"]')[0], rtol=1e-6)
    np.testing.assert_allclose(table['P(>chi2)'].iloc[1], get('a[2, "Pr(>|Chi|)"]')[0], rtol=1e-5)
    np.testing.assert_allclose(robust.covariance_, get('vcov(fr)').reshape(3, 3), rtol=1e-5)
    # the generalized leverage has trace p
    np.testing.assert_allclose(leverage(fit, X).sum(), 3)


@pytest.mark.parametrize('name', ['gaussian', 'probit', 'poisson'])
def test_glm_hatvalues_and_anova(Rinfo, name):
    rng = np.random.default_rng(2)
    n = 150
    X = rng.standard_normal((n, 3))
    eta = 0.2 + 0.5 * X[:, 0]
    family, Rfamily, y = {
        'gaussian': (sm.families.Gaussian(), 'gaussian', eta + rng.standard_normal(n)),
        'probit': (sm.families.Binomial(link=sm.families.links.Probit()), 'binomial(link="probit")',
                   rng.binomial(1, 1 / (1 + np.exp(-eta)), n)),
        'poisson': (sm.families.Poisson(), 'poisson', rng.poisson(np.exp(eta), n)),
    }[name]
    w = rng.uniform(0.5, 2, n)
    Y = pd.DataFrame({'y': y.astype(float), 'w': w})
    fit = GLM(family=family, response_id='y', weight_id='w', control=TIGHT).fit(X, Y)
    small = GLM(family=family, response_id='y', weight_id='w', exclude=[1, 2], control=TIGHT).fit(X, Y)

    _to_R(Rinfo, x0=X[:, 0], x1=X[:, 1], x2=X[:, 2], y=y, w=w)
    rpy = Rinfo['rpy']
    rpy.r(f'''suppressWarnings({{
f <- glm(y ~ x0 + x1 + x2, family={Rfamily}, weights=w, control=glm.control(epsilon=1e-14, maxit=100))
f1 <- glm(y ~ x0, family={Rfamily}, weights=w, control=glm.control(epsilon=1e-14, maxit=100))
a <- anova(f1, f, test="{'F' if name == 'gaussian' else 'Chisq'}")}})''')
    get = lambda expr: np.asarray(rpy.r(expr), float)

    np.testing.assert_allclose(leverage(fit, X), get('hatvalues(f)'), rtol=1e-6)
    table = anova(small, fit)
    np.testing.assert_allclose(table['dev_diff'].iloc[1], get('a[2, "Deviance"]')[0], rtol=1e-6)
    if name == 'gaussian':
        # least squares: the F-test, as R's anova(test="F") (and anova.lm)
        np.testing.assert_allclose(table['F'].iloc[1], get('a[2, "F"]')[0], rtol=1e-6)
        np.testing.assert_allclose(table['P(>F)'].iloc[1], get('a[2, "Pr(>F)"]')[0], rtol=1e-5)
    else:
        np.testing.assert_allclose(table['P(>chi2)'].iloc[1], get('a[2, "Pr(>Chi)"]')[0], rtol=1e-5)
    np.testing.assert_allclose(residuals(fit, 'pearson'), get('residuals(f, "pearson")'), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(residuals(fit, 'deviance'), get('residuals(f, "deviance")'), rtol=1e-6, atol=1e-8)


def test_anova_least_squares_three_models(Rinfo):
    # three nested least squares fits: R's anova.lm, F on the largest model's
    # residual degrees of freedom in every row
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 3))
    y = 0.3 + 0.5 * X[:, 0] - 0.4 * X[:, 1] + rng.standard_normal(n)
    table = anova(*[GLM(exclude=exclude).fit(X, y) for exclude in ([1, 2], [2], [])])
    _to_R(Rinfo, x0=X[:, 0], x1=X[:, 1], x2=X[:, 2], y=y)
    a = np.asarray(Rinfo['rpy'].r('a <- anova(lm(y ~ x0), lm(y ~ x0 + x1), lm(y ~ x0 + x1 + x2)); '
                                  'cbind(a$RSS, a$F, a[["Pr(>F)"]])'), float)
    np.testing.assert_allclose(table['deviance'], a[:, 0], rtol=1e-10)
    np.testing.assert_allclose(table[['F', 'P(>F)']].values[1:], a[1:, 1:], rtol=1e-10)
