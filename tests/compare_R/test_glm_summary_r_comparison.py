"""
The analysis of an unpenalized `GLM` fit (`summary_`, `dispersion_`,
`covariance_`, `predict_se`, `confidence_interval`, `prediction_interval`,
`loglike_` / `aic_` / `bic_`, `residuals`, `df_resid_`, `deviance_`) against R's
`glm`, `summary.glm`, `predict.glm` and `predict.lm`, for each family, with and
without prior weights, with an offset and without an intercept.

Where glmstar and R differ by design, the test pins down both rules:

- the reference distribution of the Wald statistics: glmstar uses t on
  `df_resid_` only for the Gaussian with identity link (estimated dispersion,
  model-based covariance), z otherwise; `summary.glm` uses t whenever the
  dispersion is estimated (also the Gamma, the inverse Gaussian and the
  Gaussian with a log link). For those, the p-values are compared with
  2 * pnorm(-|t|) computed in R from its own statistics;
- the negative binomial: glmstar fixes its dispersion at 1, while
  `summary.glm` estimates it for `MASS::negative.binomial(theta)` (it fixes it
  only for the families named binomial and poisson): R is asked for
  `dispersion=1`;
- the number of parameters of AIC / BIC: R's `logLik.glm` counts the
  dispersion as a parameter for the gaussian, Gamma and inverse.gaussian
  families, glmstar (as statsmodels) counts only the coefficients;
- the log-likelihood of the Gamma and the inverse Gaussian: R's `logLik.glm`
  (from `family$aic`) plugs in the deviance over the sum of the weights as the
  dispersion, glmstar (as statsmodels) the Pearson estimate `dispersion_`.
  The Gaussian (any link) is at the maximum likelihood estimate of its
  variance, deviance / n, in both.

Not covered: R's `residuals(type="working")` (glmstar's `residuals` has the
response, Pearson and deviance residuals only) and the Tweedie family (R's
needs statmod).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm

from glmnet import GLM
from glmnet.glm import GLMControl, confidence_interval, prediction_interval

TIGHT = GLMControl(epsnr=1e-15, mxitnr=200)
R_CONTROL = 'glm.control(epsilon=1e-15, maxit=200)'
THETA = 2.
links = sm.families.links

# name: (statsmodels family, R family, (rng, linear predictor) -> response)
FAMILIES = {
    'gaussian': (sm.families.Gaussian(), 'gaussian()',
                 lambda rng, eta: eta + rng.standard_normal(eta.shape)),
    'gaussian_log': (sm.families.Gaussian(link=links.Log()), 'gaussian(link="log")',
                     lambda rng, eta: np.exp(eta) + 0.3 * rng.standard_normal(eta.shape)),
    'logit': (sm.families.Binomial(), 'binomial()',
              lambda rng, eta: rng.binomial(1, 1 / (1 + np.exp(-eta)))),
    'probit': (sm.families.Binomial(link=links.Probit()), 'binomial(link="probit")',
               lambda rng, eta: rng.binomial(1, norm.cdf(eta))),
    'poisson': (sm.families.Poisson(), 'poisson()',
                lambda rng, eta: rng.poisson(np.exp(eta))),
    'gamma_log': (sm.families.Gamma(link=links.Log()), 'Gamma(link="log")',
                  lambda rng, eta: rng.gamma(5, np.exp(eta) / 5)),
    # the canonical link: the linear predictor 1 / mu stays positive
    'gamma_inverse': (sm.families.Gamma(), 'Gamma(link="inverse")',
                      lambda rng, eta: rng.gamma(5, 1 / (5 * eta))),
    'inverse_gaussian': (sm.families.InverseGaussian(link=links.Log()),
                         'inverse.gaussian(link="log")',
                         lambda rng, eta: rng.wald(np.exp(eta), 10)),
    'negative_binomial': (sm.families.NegativeBinomial(alpha=1 / THETA),
                          f'MASS::negative.binomial(theta={THETA})',
                          lambda rng, eta: rng.negative_binomial(THETA, THETA / (THETA + np.exp(eta)))),
}
CANONICAL = ['gaussian', 'logit', 'poisson', 'gamma_inverse']
# the families whose dispersion is estimated (by Pearson's chi-squared / df_resid)
FREE_DISPERSION = ['gaussian', 'gaussian_log', 'gamma_log', 'gamma_inverse', 'inverse_gaussian']
# R's logLik.glm counts the dispersion as a parameter for these
R_DISPERSION_DF = FREE_DISPERSION
# R's log-likelihood plugs in deviance / sum(w) as the dispersion
R_DEVIANCE_DISPERSION = ['gamma_log', 'gamma_inverse', 'inverse_gaussian']

CASES = [(name, case) for name in FAMILIES for case in ['plain', 'weights', 'offset']]
CASES += [(name, 'no_intercept') for name in ['gaussian', 'logit', 'poisson']]

n = 150
NAMES = ['x0', 'x1', 'x2']


def _data(name, case):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=NAMES)
    if name == 'gamma_inverse':
        beta0, beta = 1., np.array([0.2, -0.15, 0.])
    else:
        beta0, beta = 0.3, np.array([0.5, -0.4, 0.])
    if case == 'no_intercept':
        beta0 = 0.
    offset = 0.1 * rng.standard_normal(n) if case == 'offset' else np.zeros(n)
    eta = beta0 + X.values @ beta + offset
    y = FAMILIES[name][2](rng, eta).astype(float)
    if case != 'weights':
        w = np.ones(n)
    elif name in ['logit', 'probit']:
        # integer prior weights (numbers of trials): R's binomial log-likelihood rounds m * y
        w = rng.integers(1, 4, n).astype(float)
    else:
        w = rng.uniform(0.5, 2, n)
    Y = pd.DataFrame({'y': y, 'w': w, 'off': offset})
    start = np.hstack([beta0, beta]) if case != 'no_intercept' else beta
    return X, Y, start


def _to_R(Rinfo, X, Y, X_new, w_new):
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('Xv', X.values)
        rpy.r.assign('yv', Y['y'].values)
        rpy.r.assign('wv', Y['w'].values)
        rpy.r.assign('ov', Y['off'].values)
        rpy.r.assign('Xnewv', X_new.values)
        rpy.r.assign('wnewv', w_new)
    rpy.r(f'''d <- data.frame(Xv); colnames(d) <- c({", ".join(f'"{v}"' for v in NAMES)})
d$y <- as.vector(yv); d$w <- as.vector(wv); d$off <- as.vector(ov)
dnew <- data.frame(Xnewv); colnames(dnew) <- colnames(d)[1:3]
dnew$off <- 0; wnew <- as.vector(wnewv)''')


@pytest.mark.parametrize('name, case', CASES)
def test_glm_summary(Rinfo, name, case):
    family, Rfamily = FAMILIES[name][:2]
    X, Y, start = _data(name, case)
    fit_intercept = case != 'no_intercept'
    fit = GLM(family=family, summarize=True, response_id='y', weight_id='w', offset_id='off',
              fit_intercept=fit_intercept, control=TIGHT).fit(X, Y)
    X_new = X.iloc[:8] + 0.25
    w_new = np.linspace(0.5, 2, 8)

    _to_R(Rinfo, X, Y, X_new, w_new)
    rpy = Rinfo['rpy']
    formula = 'y ~ x0 + x1 + x2 + offset(off)' if fit_intercept else 'y ~ 0 + x0 + x1 + x2 + offset(off)'
    start_R = ', '.join(repr(float(s)) for s in start)
    # glmstar fixes the negative binomial's dispersion at 1, R would estimate it
    disp = ', dispersion=1' if name == 'negative_binomial' else ''
    rpy.r(f'''f <- suppressWarnings(glm({formula}, family={Rfamily}, data=d, weights=w,
         start=c({start_R}), control={R_CONTROL}))
s <- summary(f{disp})
pr <- predict(f, newdata=dnew, type="link", se.fit=TRUE{disp})''')
    get = lambda expr: np.asarray(rpy.r(expr), float)
    # glmstar stops on the relative change in the objective, so that the
    # coefficients are accurate to about sqrt(epsnr): Newton's method (the
    # canonical links) lands on the MLE, Fisher scoring (the others) converges
    # only linearly and is within about 1e-8 of it at epsnr=1e-15
    tol = dict(rtol=1e-8, atol=1e-9) if name in CANONICAL else dict(rtol=1e-6, atol=1e-8)

    # the coefficient table, its names in R's order
    R_names = list(rpy.r('rownames(s$coefficients)'))
    assert list(fit.summary_.index) == [{'(Intercept)': 'intercept'}.get(v, v) for v in R_names]
    coefs = get('s$coefficients')
    np.testing.assert_allclose(fit.summary_['coef'].values, coefs[:, 0], **tol)
    np.testing.assert_allclose(fit.summary_['std err'].values, coefs[:, 1], **tol)
    np.testing.assert_allclose(fit.summary_.iloc[:, 2].values, coefs[:, 2], **tol)
    if name in FREE_DISPERSION:
        assert fit._estimated_dispersion
    else:
        assert fit.dispersion_ == 1
    if name == 'negative_binomial':
        # without dispersion=1 summary.glm would estimate it
        assert get('summary(f)$dispersion')[0] != 1
    np.testing.assert_allclose(fit.dispersion_, get('s$dispersion')[0], **tol)
    np.testing.assert_allclose(fit.covariance_, get(f'vcov(f{disp})'), **tol)
    # the p-values: t on df_resid for the Gaussian with identity link, as R;
    # z for the other free-dispersion families, where R uses t
    stat = fit.summary_.columns[2]
    if name == 'gaussian':
        assert stat == 't'
        np.testing.assert_allclose(fit.summary_['P>|t|'].values, coefs[:, 3], **tol)
    elif name in FREE_DISPERSION:
        assert stat == 'z'
        assert list(rpy.r('colnames(s$coefficients)'))[3] == 'Pr(>|t|)'
        np.testing.assert_allclose(fit.summary_['P>|z|'].values,
                                   get('2 * pnorm(-abs(s$coefficients[, 3]))'), **tol)
    else:
        assert stat == 'z'
        assert list(rpy.r('colnames(s$coefficients)'))[3] == 'Pr(>|z|)'
        np.testing.assert_allclose(fit.summary_['P>|z|'].values, coefs[:, 3], **tol)
    q = get(f'qt(0.975, f$df.residual)' if name == 'gaussian' else 'qnorm(0.975)')[0]
    np.testing.assert_allclose(fit.summary_[['[0.025', '0.975]']].values,
                               coefs[:, [0]] + np.array([-1, 1]) * q * coefs[:, [1]], **tol)

    # the fit
    assert fit.df_resid_ == get('df.residual(f)')[0]
    np.testing.assert_allclose(fit.deviance_, get('deviance(f)')[0], **tol)
    np.testing.assert_allclose(fit.predict(X, offset=Y['off']), get('fitted(f)'), **tol)
    for kind in ['response', 'pearson', 'deviance']:
        np.testing.assert_allclose(fit.residuals(kind), get(f'residuals(f, type="{kind}")'), **tol)

    # the log-likelihood, AIC and BIC
    k = len(fit.params_)
    nobs = np.sum(Y['w'] > 0)
    if name in R_DEVIANCE_DISPERSION:
        # R's (from family$aic, as if the weights were frequencies) is at the
        # dispersion deviance / sum(w); glmstar's is that of the prior-weight
        # model, y_i with dispersion phi / w_i, at the Pearson phi = dispersion_
        rpy.r('''mu <- fitted(f); phi <- s$dispersion; y <- d$y; w <- d$w
dphi <- deviance(f) / sum(w)''')
        if name == 'inverse_gaussian':
            R_ll = '-sum(w) * (log(2 * pi * dphi) + 1) / 2 - 1.5 * sum(w * log(y))'
            ll = 'sum(-0.5 * log(2 * pi * phi * y^3 / w) - w * (y - mu)^2 / (2 * phi * mu^2 * y))'
        else:
            R_ll = 'sum(w * dgamma(y, 1 / dphi, scale=mu * dphi, log=TRUE))'
            ll = 'sum(dgamma(y, w / phi, scale=mu * phi / w, log=TRUE))'
        np.testing.assert_allclose(get('logLik(f)')[0], get(R_ll)[0], **tol)
        np.testing.assert_allclose(fit.loglike_, get(ll)[0], **tol)
    else:
        np.testing.assert_allclose(fit.loglike_, get('logLik(f)')[0], **tol)
        # R counts the dispersion of the Gaussian as a parameter, glmstar does not
        k_R = k + (name in R_DISPERSION_DF)
        assert get('attr(logLik(f), "df")')[0] == k_R
        np.testing.assert_allclose(fit.aic_, get('AIC(f)')[0] - 2 * (k_R - k), **tol)
        np.testing.assert_allclose(fit.bic_, get('BIC(f)')[0] - np.log(nobs) * (k_R - k), **tol)
    np.testing.assert_allclose(fit.aic_, -2 * fit.loglike_ + 2 * k, **tol)
    np.testing.assert_allclose(fit.bic_, -2 * fit.loglike_ + np.log(nobs) * k, **tol)

    # standard errors and confidence intervals of the linear predictor at new
    # rows (without the offset), as predict.glm(se.fit=TRUE) at the dispersion of summary.glm
    se = fit.predict_se(X_new)
    assert list(se.index) == list(X_new.index)
    np.testing.assert_allclose(se.values, get('pr$se.fit'), **tol)
    for alpha in [0.05, 0.1]:
        ci = confidence_interval(fit, X_new, alpha=alpha)
        q = get(f'qt({1 - alpha / 2}, f$df.residual)' if name == 'gaussian'
                else f'qnorm({1 - alpha / 2})')[0]
        np.testing.assert_allclose(ci['estimate'].values, get('pr$fit'), **tol)
        np.testing.assert_allclose(ci['lower'].values, get(f'pr$fit - {float(q)!r} * pr$se.fit'), **tol)
        np.testing.assert_allclose(ci['upper'].values, get(f'pr$fit + {float(q)!r} * pr$se.fit'), **tol)

    # prediction intervals for the Gaussian with identity link, as predict.lm,
    # the variance of a new response sigma^2 / w at its prior weight w
    if name == 'gaussian':
        rpy.r(f'l <- lm({formula}, data=d, weights=w)')
        # without weights= R takes the new responses' prior weights to be 1
        for alpha, w in [(0.05, None), (0.1, w_new)]:
            Rw = '' if w is None else ', weights=wnew'
            pi_R = get(f'suppressWarnings(predict.lm(l, newdata=dnew, interval="prediction", '
                       f'level={1 - alpha}{Rw}))')
            pi = prediction_interval(fit, X_new, alpha=alpha, weights=w)
            np.testing.assert_allclose(pi[['estimate', 'lower', 'upper']].values, pi_R, **tol)
    else:
        with pytest.raises(ValueError):
            prediction_interval(fit, X_new)

