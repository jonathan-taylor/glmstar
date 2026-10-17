"""
The standard analysis of an unpenalized GLM fit (coefficient table, intervals,
log-likelihood, residuals, prediction standard errors, leverage, analysis of
deviance) against statsmodels, with weights as prior weights (``var_weights``).
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import statsmodels.api as sm
from scipy.stats import chi2

from glmnet import GLM
from glmnet.glm import GLMControl, BinomialGLM, anova, prediction_interval

TIGHT = GLMControl(epsnr=1e-12, mxitnr=100)
rng = np.random.default_rng(0)
n, p = 200, 4
X = rng.standard_normal((n, p))
W = rng.uniform(0.5, 2, n)
OFFSET = 0.2 * rng.standard_normal(n)

FAMILIES = {
    'gaussian': (sm.families.Gaussian(), lambda eta: eta + rng.standard_normal(n)),
    'logit': (sm.families.Binomial(), lambda eta: rng.binomial(1, 1 / (1 + np.exp(-eta)))),
    'probit': (sm.families.Binomial(link=sm.families.links.Probit()),
               lambda eta: rng.binomial(1, 1 / (1 + np.exp(-eta)))),
    'poisson': (sm.families.Poisson(), lambda eta: rng.poisson(np.exp(eta))),
    # families with a free dispersion, estimated by Pearson's chi-squared over df_resid
    'gaussian_log': (sm.families.Gaussian(link=sm.families.links.Log()),
                     lambda eta: np.exp(eta) + 0.3 * rng.standard_normal(n)),
    'gamma': (sm.families.Gamma(link=sm.families.links.Log()),
              lambda eta: rng.gamma(5, np.exp(eta) / 5)),
    'inverse_gaussian': (sm.families.InverseGaussian(link=sm.families.links.Log()),
                         lambda eta: rng.wald(np.exp(eta), 10)),
}
FIXED_DISPERSION = ['gaussian', 'logit', 'probit', 'poisson']
CASES = ['plain', 'weights', 'offset', 'exclude', 'no_intercept', 'sparse']


def _fits(name, case, sandwich=False):
    family, draw = FAMILIES[name]
    y = draw(0.3 + 0.5 * X[:, 0] - 0.4 * X[:, 1]).astype(float)
    w = W if case == 'weights' else np.ones(n)
    offset = OFFSET if case == 'offset' else None
    exclude = [2] if case == 'exclude' else []
    fit_intercept = case != 'no_intercept'
    Y = pd.DataFrame({'y': y, 'w': w, 'offset': OFFSET if offset is not None else 0.})
    X_fit = scipy.sparse.csc_array(X) if case == 'sparse' else X
    g = GLM(family=family, summarize=True, response_id='y', weight_id='w', offset_id='offset',
            exclude=exclude, fit_intercept=fit_intercept, sandwich=sandwich,
            control=TIGHT).fit(X_fit, Y)
    kept = [j for j in range(p) if j not in exclude]
    D = sm.add_constant(X[:, kept]) if fit_intercept else X[:, kept]
    model = sm.GLM(y, D, family=family, var_weights=w, offset=offset)
    if sandwich:
        # HC0 with the expected information as its bread (statsmodels' IRLS uses
        # the observed for a non-canonical link), as R's sandwich(glm)
        s = model.fit(method='newton', optim_hessian='eim', tol=1e-12, maxiter=100, cov_type='HC0')
    else:
        s = model.fit(tol=1e-12)
    return g, s, X_fit, D, offset


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('name', list(FAMILIES))
def test_against_statsmodels(name, case):
    g, s, X_fit, D, offset = _fits(name, case)
    # the non-canonical log links of the free-dispersion families converge less tightly
    tol = dict(rtol=1e-6, atol=1e-8) if name in FIXED_DISPERSION else dict(rtol=1e-4, atol=1e-6)

    np.testing.assert_allclose(g.params_.values, s.params, **tol)
    np.testing.assert_allclose(g.summary_['std err'].values, s.bse, **tol)
    np.testing.assert_allclose(g.summary_.iloc[:, 2].values, s.tvalues, **tol)
    np.testing.assert_allclose(g.dispersion_, s.scale, **tol)
    if name != 'gaussian':      # the Gaussian's p-values and intervals use t, statsmodels' GLM z
        np.testing.assert_allclose(g.summary_.iloc[:, 3].values, s.pvalues, **tol)
        np.testing.assert_allclose(g.summary_[['[0.025', '0.975]']].values, s.conf_int(), **tol)

    if name == 'gaussian_log':
        # the Gaussian's at the MLE of its variance for every link, as R's logLik.glm
        # (statsmodels takes the MLE only for the identity link)
        mle = s.model.family.loglike(s.model.endog, s.mu, var_weights=s.model.var_weights,
                                     scale=np.sum(s.model.var_weights * (s.model.endog - s.mu)**2) / n)
        np.testing.assert_allclose(g.loglike_, mle, **tol)
    else:
        np.testing.assert_allclose(g.loglike_, s.llf, **tol)
        np.testing.assert_allclose(g.aic_, s.aic, **tol)
        np.testing.assert_allclose(g.bic_, s.bic_llf, **tol)

    np.testing.assert_allclose(g.predict(X_fit, offset=offset), s.fittedvalues, **tol)
    for kind, expected in [('response', s.resid_response), ('pearson', s.resid_pearson),
                           ('deviance', s.resid_deviance)]:
        np.testing.assert_allclose(g.residuals(kind), expected, **tol)

    rows = slice(0, 7)
    X_new = X_fit[rows] if not scipy.sparse.issparse(X_fit) else X_fit[[0, 1, 2, 3, 4, 5, 6]]
    off_new = None if offset is None else offset[rows]
    linear = s.get_prediction(D[rows], offset=off_new, which='linear')
    np.testing.assert_allclose(g.predict_se(X_new), linear.se, **tol)
    if name != 'gaussian':
        ci = g.conf_int(X_new)
        linear = s.get_prediction(D[rows], which='linear')
        np.testing.assert_allclose(ci['estimate'], linear.predicted, **tol)
        np.testing.assert_allclose(ci[['lower', 'upper']].values, linear.conf_int(), **tol)

    # the expected information, as R's hatvalues (statsmodels defaults to the observed)
    np.testing.assert_allclose(g.leverage(X_fit), s.get_hat_matrix_diag(observed=False), **tol)


@pytest.mark.parametrize('case', ['plain', 'weights', 'offset', 'exclude', 'sparse'])
@pytest.mark.parametrize('name', list(FAMILIES))
def test_sandwich(name, case):
    g, s, X_fit, D, offset = _fits(name, case, sandwich=True)
    tol = dict(rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(g.covariance_, s.cov_params(), **tol)
    np.testing.assert_allclose(g.summary_['std err'].values, s.bse, **tol)
    np.testing.assert_allclose(g.predict_se(X_fit[:5]), s.get_prediction(D[:5], which='linear').se, **tol)
    # the leverage does not change
    np.testing.assert_allclose(g.leverage(X_fit), s.get_hat_matrix_diag(observed=False), **tol)
    with pytest.warns(UserWarning, match='sandwich'):
        anova(g, g)


@pytest.mark.parametrize('name', list(FAMILIES))
def test_anova(name):
    family, draw = FAMILIES[name]
    y = draw(0.3 + 0.5 * X[:, 0] - 0.4 * X[:, 1]).astype(float)
    small, big = (GLM(family=family, exclude=exclude, control=TIGHT).fit(X, y)
                  for exclude in ([1, 2, 3], []))
    table = anova(small, big)
    s_small = sm.GLM(y, sm.add_constant(X[:, :1]), family=family).fit(tol=1e-12)
    s_big = sm.GLM(y, sm.add_constant(X), family=family).fit(tol=1e-12)
    dev = s_small.deviance - s_big.deviance
    np.testing.assert_allclose(table['df_resid'], [n - 2, n - 5])
    np.testing.assert_allclose(table['dev_diff'].iloc[1], dev, rtol=1e-6)
    np.testing.assert_allclose(table['chi2'].iloc[1], dev / s_big.scale, rtol=1e-6)
    np.testing.assert_allclose(table['P(>chi2)'].iloc[1], chi2.sf(dev / s_big.scale, 3), rtol=1e-5)


def test_gaussian_dispersion_is_estimated():
    # GLM.fit used to fix the Gaussian's dispersion at 1
    y = 3 * rng.standard_normal(n)
    g = GLM(summarize=True).fit(X, y)
    s = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(g.dispersion_, s.scale)
    np.testing.assert_allclose(g.summary_['std err'], s.bse)
    np.testing.assert_allclose(g.summary_['P>|t|'], s.pvalues)
    assert g.df_resid_ == s.df_resid
    # a given dispersion is used as it is
    assert GLM(summarize=True).fit(X, y, dispersion=2.).dispersion_ == 2.


def test_t_or_z():
    # t only for the Gaussian with identity link, an estimated dispersion and sandwich=False
    y = 3 + X[:, 0] + rng.standard_normal(n)
    stat = lambda g: g.summary_.columns[2]
    assert stat(GLM(summarize=True).fit(X, y)) == 't'
    assert stat(GLM(summarize=True).fit(X, y, dispersion=2.)) == 'z'
    assert stat(GLM(summarize=True, sandwich=True).fit(X, y)) == 'z'
    for family in [sm.families.Gaussian(link=sm.families.links.Log()),
                   sm.families.Gamma(link=sm.families.links.Log())]:
        assert stat(GLM(family=family, summarize=True).fit(X, np.exp(y / 3))) == 'z'
    # a given dispersion is used as it is, also for the Gamma
    g = GLM(family=sm.families.Gamma(link=sm.families.links.Log()),
            summarize=True).fit(X, np.exp(y / 3), dispersion=0.5)
    assert g.dispersion_ == 0.5


@pytest.mark.parametrize('weighted', [False, True])
def test_prediction_interval(weighted):
    # as statsmodels' OLS / WLS get_prediction(...).summary_frame()'s obs_ci
    y = 3 + X[:, 0] + rng.standard_normal(n) / np.sqrt(W if weighted else 1)
    w = W if weighted else np.ones(n)
    g = GLM(summarize=True, response_id='y', weight_id='w').fit(X, pd.DataFrame({'y': y, 'w': w}))
    s = sm.WLS(y, sm.add_constant(X), weights=w).fit()
    X_new, w_new = X[:6], W[:6] if weighted else None
    frame = s.get_prediction(sm.add_constant(X_new), weights=w_new).summary_frame(alpha=0.1)
    pi = prediction_interval(g, X_new, alpha=0.1, weights=w_new)
    np.testing.assert_allclose(pi['estimate'], frame['mean'])
    np.testing.assert_allclose(pi[['lower', 'upper']].values, frame[['obs_ci_lower', 'obs_ci_upper']].values)
    # the confidence intervals are those of the mean
    ci = g.conf_int(X_new, alpha=0.1)
    np.testing.assert_allclose(ci[['lower', 'upper']].values, frame[['mean_ci_lower', 'mean_ci_upper']].values)


def test_prediction_interval_only_gaussian():
    y = 3 + X[:, 0] + rng.standard_normal(n)
    for g in [GLM(summarize=True, sandwich=True).fit(X, y),
              GLM(summarize=True).fit(X, y, dispersion=1.),
              GLM(family=sm.families.Poisson(), summarize=True).fit(X, rng.poisson(2, n))]:
        with pytest.raises(ValueError, match='prediction intervals'):
            prediction_interval(g, X[:3])


def test_exclude():
    # excluded variables are left out of the fit, the table and df_resid
    y = X[:, 0] + rng.standard_normal(n)
    g = GLM(summarize=True, exclude=[1, 3]).fit(X, y)
    s = sm.OLS(y, sm.add_constant(X[:, [0, 2]])).fit()
    np.testing.assert_allclose(g.coef_[[1, 3]], 0)
    np.testing.assert_allclose(g.params_.values, s.params)
    assert list(g.summary_.index) == ['intercept', 'X0', 'X2']
    np.testing.assert_allclose(g.summary_['std err'], s.bse)
    assert g.df_resid_ == s.df_resid


def test_islp_design_and_summarize():
    # a ModelSpec-like DataFrame with a constant 'intercept' column, and ISLP's summarize
    df = pd.DataFrame(X, columns=list('abcd'))
    D = pd.concat([pd.DataFrame({'intercept': np.ones(n)}), df], axis=1)
    y = X[:, 0] + rng.standard_normal(n)
    with pytest.warns(UserWarning, match='intercept'):
        g = GLM(summarize=True).fit(D, y)
    s = sm.OLS(y, D).fit()
    assert list(g.params_.index) == list(s.params.index)
    np.testing.assert_allclose(g.params_, s.params)
    np.testing.assert_allclose(g.predict(D), s.predict(D))
    np.testing.assert_allclose(g.conf_int(D[:5])[['lower', 'upper']].values,
                               s.get_prediction(D[:5]).conf_int())
    np.testing.assert_allclose(g.summary_[['[0.025', '0.975]']].values, s.conf_int())

    summarize = pytest.importorskip('ISLP.models').summarize
    for conf_int in [False, True]:
        ours, theirs = summarize(g, conf_int=conf_int), summarize(s, conf_int=conf_int)
        # statsmodels labels the intervals with alpha / 2 = 0.025000000000000022
        theirs.columns = ours.columns
        pd.testing.assert_frame_equal(ours, theirs)


@pytest.mark.parametrize('sandwich', [False, True])
def test_pandas_outputs(sandwich):
    # the outputs at the rows of a new X keep the labels of a DataFrame X
    index = pd.Index([f'obs{i}' for i in range(n)])
    D = pd.DataFrame(X, columns=list('abcd'), index=index)
    y = 3 + X[:, 0] + rng.standard_normal(n)
    g = GLM(summarize=True, sandwich=sandwich).fit(D, y)
    new = D.iloc[[3, 1, 4]]
    se = g.predict_se(new)
    assert isinstance(se, pd.Series) and list(se.index) == list(new.index)
    for out in [g.conf_int(new)] + ([] if sandwich else [prediction_interval(g, new)]):
        assert isinstance(out, pd.DataFrame) and list(out.index) == list(new.index)
    # an ndarray X: a RangeIndex, and the same values
    h = GLM(summarize=True, sandwich=sandwich).fit(X, y)
    assert h.predict_se(X[:3]).index.equals(pd.RangeIndex(3))
    assert h.conf_int(X[:3]).index.equals(pd.RangeIndex(3))
    np.testing.assert_allclose(h.predict_se(X[:3]), g.predict_se(D[:3]))
    # residuals, leverage and covariance_ stay ndarrays
    assert all(isinstance(out, np.ndarray)
               for out in [g.residuals(), g.leverage(D), g.covariance_])


@pytest.mark.parametrize('family', [None, sm.families.Binomial()])
def test_binomial_glm_predict_class(family):
    y = np.where(rng.binomial(1, 1 / (1 + np.exp(-X[:, 0]))), 'up', 'down')
    g = (BinomialGLM() if family is None else BinomialGLM(family=family)).fit(X, y)
    pred = g.predict(X)
    assert set(pred) <= {'up', 'down'}
    np.testing.assert_array_equal(pred == 'up', g.predict(X, prediction_type='response') > 0.5)


@pytest.mark.parametrize('sparse', [False, True])
def test_intercept_only(sparse):
    # every variable excluded, or a ModelSpec design of the intercept alone
    y = rng.poisson(2, n).astype(float)
    family = sm.families.Poisson()
    s = sm.GLM(y, np.ones((n, 1)), family=family).fit(tol=1e-12)
    X_fit = scipy.sparse.csc_array(X) if sparse else X
    D = pd.DataFrame({'intercept': np.ones(n)})
    for g in [GLM(family=family, summarize=True, exclude=list(range(p)), control=TIGHT).fit(X_fit, y),
              GLM(family=family, summarize=True, control=TIGHT).fit(D, y)]:
        assert list(g.params_.index) == ['intercept']
        np.testing.assert_allclose(g.params_.values, s.params)
        np.testing.assert_allclose(g.summary_['std err'].values, s.bse)
        np.testing.assert_allclose(g.aic_, s.aic)
        assert g.df_resid_ == n - 1
    np.testing.assert_allclose(g.predict(D), s.predict(np.ones((n, 1))))
