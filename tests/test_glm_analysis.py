"""
The standard analysis of an unpenalized GLM fit (coefficient table, intervals,
log-likelihood, residuals, prediction standard errors, leverage, analysis of
deviance) against statsmodels, with weights as prior weights (``var_weights``).
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import statsmodels.api as sm
from scipy.stats import chi2, f as fdist

from glmnet import GLM
from glmnet.glm import (GLMControl, BinomialGLM, anova, confidence_interval,
                        leverage, prediction_interval, residuals, summarize)

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
        np.testing.assert_allclose(residuals(g, kind), expected, **tol)

    rows = slice(0, 7)
    X_new = X_fit[rows] if not scipy.sparse.issparse(X_fit) else X_fit[[0, 1, 2, 3, 4, 5, 6]]
    off_new = None if offset is None else offset[rows]
    linear = s.get_prediction(D[rows], offset=off_new, which='linear')
    np.testing.assert_allclose(g.predict_se(X_new), linear.se, **tol)
    if name != 'gaussian':
        ci = confidence_interval(g, X_new)
        linear = s.get_prediction(D[rows], which='linear')
        np.testing.assert_allclose(ci['estimate'], linear.predicted, **tol)
        np.testing.assert_allclose(ci[['lower', 'upper']].values, linear.conf_int(), **tol)

    # the expected information, as R's hatvalues (statsmodels defaults to the observed)
    np.testing.assert_allclose(leverage(g, X_fit), s.get_hat_matrix_diag(observed=False), **tol)


@pytest.mark.parametrize('case', ['plain', 'weights', 'offset', 'exclude', 'sparse'])
@pytest.mark.parametrize('name', list(FAMILIES))
def test_sandwich(name, case):
    g, s, X_fit, D, offset = _fits(name, case, sandwich=True)
    tol = dict(rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(g.covariance_, s.cov_params(), **tol)
    np.testing.assert_allclose(g.summary_['std err'].values, s.bse, **tol)
    np.testing.assert_allclose(g.predict_se(X_fit[:5]), s.get_prediction(D[:5], which='linear').se, **tol)
    # the leverage does not change
    np.testing.assert_allclose(leverage(g, X_fit), s.get_hat_matrix_diag(observed=False), **tol)
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
    if name == 'gaussian':
        # least squares: the F-test (anova_lm), not chi-squared
        assert 'chi2' not in table
        F = dev / 3 / s_big.scale
        np.testing.assert_allclose(table['F'].iloc[1], F, rtol=1e-6)
        np.testing.assert_allclose(table['P(>F)'].iloc[1], fdist.sf(F, 3, n - 5), rtol=1e-5)
    else:
        assert 'F' not in table
        np.testing.assert_allclose(table['chi2'].iloc[1], dev / s_big.scale, rtol=1e-6)
        np.testing.assert_allclose(table['P(>chi2)'].iloc[1], chi2.sf(dev / s_big.scale, 3), rtol=1e-5)


def test_anova_least_squares():
    # three nested least squares fits: statsmodels' anova_lm (F against the
    # largest model's residual mean square)
    from statsmodels.stats.anova import anova_lm
    y = 0.3 + 0.5 * X[:, 0] - 0.4 * X[:, 1] + rng.standard_normal(n)
    fits = [GLM(exclude=exclude).fit(X, y) for exclude in ([1, 2, 3], [2, 3], [])]
    table = anova(*fits)
    ols = [sm.OLS(y, sm.add_constant(X[:, :k])).fit() for k in (1, 2, 4)]
    expected = anova_lm(*ols)
    np.testing.assert_allclose(table['deviance'], expected['ssr'], rtol=1e-10)
    np.testing.assert_allclose(table['df_diff'].iloc[1:], expected['df_diff'].iloc[1:])
    np.testing.assert_allclose(table['F'].iloc[1:], expected['F'].iloc[1:], rtol=1e-8)
    # the p-values on the largest model's residual df, as R's anova.lm
    # (statsmodels uses each row's own df_resid, which agrees for the last row)
    np.testing.assert_allclose(table['P(>F)'].iloc[1:],
                               fdist.sf(table['F'].iloc[1:], table['df_diff'].iloc[1:], n - 5),
                               rtol=1e-10)
    np.testing.assert_allclose(table['P(>F)'].iloc[-1], expected['Pr(>F)'].iloc[-1], rtol=1e-8)
    # a given dispersion, the sandwich covariance or a non-identity link: chi-squared
    for kwargs, fit_kwargs in [({}, {'dispersion': 1.}), ({'sandwich': True}, {}),
                               ({'family': sm.families.Gaussian(link=sm.families.links.Log())}, {})]:
        yk = np.exp(0.1 * y) if 'family' in kwargs else y
        small, big = (GLM(exclude=exclude, **kwargs).fit(X, yk, **fit_kwargs)
                      for exclude in ([1, 2, 3], []))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)   # sandwich
            table = anova(small, big)
        assert 'chi2' in table and 'F' not in table


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
    ci = confidence_interval(g, X_new, alpha=0.1)
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
    # a ModelSpec-like DataFrame with a constant 'intercept' column, and summarize
    df = pd.DataFrame(X, columns=list('abcd'))
    D = pd.concat([pd.DataFrame({'intercept': np.ones(n)}), df], axis=1)
    y = X[:, 0] + rng.standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # the 'intercept' column is dropped silently
        g = GLM(summarize=True).fit(D, y)
    s = sm.OLS(y, D).fit()
    assert list(g.params_.index) == list(s.params.index)
    np.testing.assert_allclose(g.params_, s.params)
    np.testing.assert_allclose(g.predict(D), s.predict(D))
    np.testing.assert_allclose(confidence_interval(g, D[:5])[['lower', 'upper']].values,
                               s.get_prediction(D[:5]).conf_int())
    np.testing.assert_allclose(g.summary_[['[0.025', '0.975]']].values, s.conf_int())

    # the table ISLP's summarize gives for statsmodels, unrounded
    expected = pd.DataFrame({'coef': s.params, 'std err': s.bse, 't': s.tvalues,
                             'P>|t|': s.pvalues})
    pd.testing.assert_frame_equal(summarize(g), expected, rtol=1e-8)
    for kwargs, alpha in [({'conf_int': True}, 0.05), ({'level': 0.9}, 0.1)]:
        table = summarize(g, **kwargs)
        bounds = [f'[{alpha / 2:g}', f'{1 - alpha / 2:g}]']
        assert list(table.columns) == list(expected.columns) + bounds
        np.testing.assert_allclose(table[bounds].values, s.conf_int(alpha=alpha), rtol=1e-8)


def test_summarize_full(capsys):
    # full=True prints a description of the fit above the table; the default doesn't
    y = X[:, 0] + rng.standard_normal(n)
    g = GLM(summarize=True).fit(X, y)
    table = summarize(g)
    assert capsys.readouterr().out == ''
    pd.testing.assert_frame_equal(summarize(g, full=True), table)
    out = capsys.readouterr().out
    for label, value in [('Df residuals:', f'{g.df_resid_:g}'), ('Deviance:', f'{g.deviance_:.4f}'),
                         ('Null deviance:', f'{g.null_deviance_:.4f}'), ('AIC:', f'{g.aic_:.4f}'),
                         ('Covariance:', 'model')]:
        assert label in out and value in out
    # without summarize=True the table is computed; a binomial fit has z statistics
    yb = rng.binomial(1, 0.5, n)
    b = GLM(family=sm.families.Binomial()).fit(X, yb)
    assert b.summary_ is None
    assert list(summarize(b).columns) == ['coef', 'std err', 'z', 'P>|z|']
    s = sm.GLM(yb, sm.add_constant(X), family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(summarize(b)['std err'], s.bse, rtol=1e-6)


@pytest.mark.parametrize('sandwich', [False, True])
def test_pandas_outputs(sandwich):
    # outputs indexed by the features are labelled by their names; outputs at the
    # rows of a DataFrame X are labelled by its index, at the rows of an array are arrays
    index = pd.Index([f'obs{i}' for i in range(n)])
    D = pd.DataFrame(X, columns=list('abcd'), index=index)
    y = 3 + X[:, 0] + rng.standard_normal(n)
    g = GLM(summarize=True, sandwich=sandwich).fit(D, y)
    h = GLM(summarize=True, sandwich=sandwich).fit(X, y)
    names = ['intercept', 'a', 'b', 'c', 'd']

    # covariance_ is an array, labelled by feature_names_ (the names of params_)
    for fit in [g, h]:
        C = fit.covariance_
        assert isinstance(C, np.ndarray) and C.shape == (5, 5)
        assert isinstance(fit.feature_names_, np.ndarray)
        assert list(fit.feature_names_) == list(fit.params_.index) == list(fit.summary_.index)
        np.testing.assert_allclose(np.sqrt(np.diag(C)), fit.summary_['std err'])
    assert list(g.feature_names_) == names
    assert list(h.feature_names_) == ['intercept', 'X0', 'X1', 'X2', 'X3']
    nointercept = GLM(summarize=True, fit_intercept=False, exclude=[2]).fit(D, y)
    assert list(nointercept.feature_names_) == ['a', 'b', 'd']
    assert nointercept.covariance_.shape == (3, 3)

    new = D.iloc[[3, 1, 4]]
    se = g.predict_se(new)
    assert isinstance(se, pd.Series) and list(se.index) == list(new.index)
    frames = [confidence_interval(g, new)] + ([] if sandwich else [prediction_interval(g, new)])
    for out in frames:
        assert isinstance(out, pd.DataFrame) and list(out.index) == list(new.index)
        assert list(out.columns) == ['estimate', 'lower', 'upper']
    # an array X_new (to the fit to an array): an array for predict_se, with the same values
    assert isinstance(h.predict_se(X[[3, 1, 4]]), np.ndarray)
    np.testing.assert_allclose(h.predict_se(X[[3, 1, 4]]), se)
    arrays = [confidence_interval(h, X[[3, 1, 4]])] + ([] if sandwich else [prediction_interval(h, X[[3, 1, 4]])])
    # the intervals are always DataFrames (their columns are named), on a RangeIndex here
    for out, frame in zip(arrays, frames):
        assert isinstance(out, pd.DataFrame) and out.index.equals(pd.RangeIndex(3))
        assert list(out.columns) == ['estimate', 'lower', 'upper']
        np.testing.assert_allclose(out.values, frame.values)

    # residuals and leverage: on the index of a DataFrame X of the fit, else arrays
    for kind in ['response', 'pearson', 'deviance']:
        r = residuals(g, kind)
        assert isinstance(r, pd.Series) and r.index.equals(index) and r.name == kind
        assert isinstance(residuals(h, kind), np.ndarray)
        np.testing.assert_allclose(r, residuals(h, kind))
    lev = leverage(g, D)
    assert isinstance(lev, pd.Series) and lev.index.equals(index)
    assert isinstance(leverage(h, X), np.ndarray)
    np.testing.assert_allclose(lev, leverage(h, X))


def test_cox_pandas_outputs():
    from glmnet.cox import CoxLM, CoxFamilySpec
    index = pd.Index([f'obs{i}' for i in range(n)])
    D = pd.DataFrame(X, columns=list('abcd'), index=index)
    Y = pd.DataFrame({'time': rng.exponential(np.exp(-0.5 * X[:, 0])),
                      'status': rng.binomial(1, 0.7, n)}, index=index)
    fam = lambda: CoxFamilySpec(event_data=Y, event_id='time', status_id='status')
    g = CoxLM(family=fam(), summarize=True).fit(D, Y)
    h = CoxLM(family=fam(), summarize=True).fit(X, Y)
    for fit in [g, h]:
        assert list(fit.feature_names_) == list(fit.params_.index)
        assert isinstance(fit.covariance_, np.ndarray) and fit.covariance_.shape == (4, 4)
    assert list(g.feature_names_) == list('abcd')
    for kind in ['martingale', 'deviance']:
        r = residuals(g, kind)
        assert isinstance(r, pd.Series) and r.index.equals(index)
        assert isinstance(residuals(h, kind), np.ndarray)
        np.testing.assert_allclose(r, residuals(h, kind))
    assert leverage(g, D).index.equals(index) and isinstance(leverage(h, X), np.ndarray)
    assert list(confidence_interval(g, D[:4]).index) == list(index[:4])
    assert isinstance(g.predict_se(D[:4]), pd.Series) and isinstance(h.predict_se(X[:4]), np.ndarray)


def test_feature_names_checked_at_predict():
    # as sklearn: the columns of a DataFrame X at predict must be those of fit
    D = pd.DataFrame(X, columns=list('abcd'))
    y = 3 + X[:, 0] + rng.standard_normal(n)
    g = GLM(summarize=True).fit(D, y)
    calls = [g.predict, g.predict_se, lambda Z: confidence_interval(g, Z),
             lambda Z: prediction_interval(g, Z)]
    bad = {'reordered': (D[['b', 'a', 'c', 'd']], 'same order'),
           'renamed': (D.rename(columns={'a': 'z'}), 'unseen at fit time'),
           'extra': (D.assign(e=1.), 'unseen at fit time'),
           'missing': (D[['a', 'b', 'c']], 'yet now missing')}
    for Z, match in bad.values():
        for call in calls:
            with pytest.raises(ValueError, match=match):
                call(Z)
    with pytest.raises(ValueError, match='feature names should match'):
        leverage(g, D[['b', 'a', 'c', 'd']])
    # an array where fit had names, and names where fit had none: warnings
    with pytest.warns(UserWarning, match='does not have valid feature names'):
        g.predict(X)
    h = GLM(summarize=True).fit(X, y)
    with pytest.warns(UserWarning, match='fitted without feature names'):
        h.predict(D)
    # the columns of fit, in order, are fine
    np.testing.assert_allclose(g.predict(D), h.predict(X))


def test_feature_names_modelspec():
    # ISLP's ModelSpec: the 'intercept' column is dropped at fit and at predict
    ModelSpec = pytest.importorskip('ISLP.models').ModelSpec
    df = pd.DataFrame(X, columns=list('abcd'))
    y = 3 + X[:, 0] + rng.standard_normal(n)
    spec = ModelSpec(['a', 'b'], intercept=True).fit(df[:150])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # the 'intercept' column is dropped silently
        g = GLM(summarize=True).fit(spec.transform(df[:150]), y[:150])
    new = spec.transform(df[150:])
    assert 'intercept' in new.columns
    pred = g.predict(new)
    assert pred.shape == (50,)
    ci = confidence_interval(g, new)
    assert list(ci.index) == list(new.index)
    assert list(g.feature_names_) == ['intercept', 'a', 'b']


@pytest.mark.parametrize('kind', ['gaussian', 'poisson', 'binomial'])
def test_predict_is_ndarray(kind):
    # as sklearn's estimators, predict returns an array even for a DataFrame X
    # (whose index is not kept); residuals, predict_se etc. are Series
    rng_ = np.random.default_rng(1)
    D = pd.DataFrame(rng_.standard_normal((50, 2)), columns=['a', 'b'],
                     index=np.arange(100, 150))
    eta = 0.5 * D['a']
    if kind == 'gaussian':
        y, g = eta + rng_.standard_normal(50), GLM()
    elif kind == 'poisson':
        y, g = rng_.poisson(np.exp(eta)), GLM(family=sm.families.Poisson())
    else:
        y, g = rng_.binomial(1, 1 / (1 + np.exp(-eta))), BinomialGLM()
    g.fit(D, y)
    for prediction_type in ['response', 'link']:
        pred = g.predict(D, prediction_type=prediction_type)
        assert type(pred) is np.ndarray
        assert pred.shape == (50,)
    if kind == 'binomial':
        pred = g.predict(D)
        assert type(pred) is np.ndarray
        assert set(pred) <= set(g.classes_)


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


@pytest.mark.parametrize('family', [sm.families.Gaussian(), sm.families.Gamma(sm.families.links.Log()),
                                    sm.families.InverseGaussian(sm.families.links.Log()),
                                    sm.families.Poisson(), sm.families.Binomial()],
                         ids=['gaussian', 'gamma', 'inverse_gaussian', 'poisson', 'binomial'])
def test_loglike_zero_weights(family):
    # observations of weight 0 are not observations: the log-likelihood and
    # AIC are those of the fit without them (finite, no warning)
    rng = np.random.default_rng(3)
    n = 120
    X = rng.standard_normal((n, 2))
    eta = 0.3 + X @ np.array([0.4, -0.3])
    if isinstance(family, sm.families.Binomial):
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    elif isinstance(family, sm.families.Poisson):
        y = rng.poisson(np.exp(eta)).astype(float)
    elif isinstance(family, sm.families.Gaussian):
        y = eta + rng.standard_normal(n)
    else:
        y = np.exp(eta) * rng.gamma(4, 1 / 4, n)
    w = rng.uniform(0.5, 2, n)
    w[:15] = 0
    D = pd.DataFrame({'y': y, 'w': w})
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        G = GLM(family=family, response_id='y', weight_id='w').fit(X, D)
    keep = w > 0
    H = GLM(family=family, response_id='y', weight_id='w').fit(X[keep], D[keep])
    assert np.isfinite(G.loglike_)
    np.testing.assert_allclose(G.loglike_, H.loglike_, rtol=1e-8)
    np.testing.assert_allclose(G.aic_, H.aic_, rtol=1e-8)


@pytest.mark.parametrize('family', [sm.families.Gaussian(), sm.families.Binomial()],
                         ids=['gaussian', 'binomial'])
def test_islp_empty_design(family):
    # ISLP's empty design (a ModelSpec with no terms, the null model of a
    # stepwise search) is a single column of zeros named 'zero': the
    # intercept-only model, as a lone 'intercept' column, without a warning
    rng = np.random.default_rng(5)
    n = 80
    y = (rng.binomial(1, 0.3, n) if isinstance(family, sm.families.Binomial)
         else rng.standard_normal(n) + 2.).astype(float)
    Z = pd.DataFrame({'zero': np.zeros(n)})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        G = GLM(family=family).fit(Z, y)
        pred = G.predict(Z.iloc[:5])
    I = GLM(family=family).fit(pd.DataFrame({'intercept': np.ones(n)}), y)
    np.testing.assert_allclose(G.coef_, 0)
    np.testing.assert_allclose(G.intercept_, I.intercept_, rtol=1e-10)
    np.testing.assert_allclose(pred, np.full(5, family.link.inverse(I.intercept_)), rtol=1e-10)
    np.testing.assert_allclose(G.deviance_, G.null_deviance_, rtol=1e-10)
    # other constant columns still warn: zeros next to other columns, or zeros
    # not named 'zero'
    X = pd.DataFrame({'zero': np.zeros(n), 'x': rng.standard_normal(n)})
    with pytest.warns(UserWarning, match='excluding the constant column'):
        GLM(family=family).fit(X, y)
    with pytest.warns(UserWarning, match='excluding the constant column'):
        GLM(family=family).fit(pd.DataFrame({'nothing': np.zeros(n)}), y)


def test_islp_empty_design_cox():
    # the null model of a Cox stepwise search: no warning, coefficient 0
    from glmnet.cox import CoxFamily, CoxLM
    rng = np.random.default_rng(6)
    n = 60
    Y = pd.DataFrame({'event': rng.exponential(1, n), 'status': rng.binomial(1, 0.7, n)})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        L = CoxLM(family=CoxFamily()).fit(pd.DataFrame({'zero': np.zeros(n)}), Y)
    np.testing.assert_allclose(L.coef_, 0)


def test_islp_stepwise_no_warning():
    # ISLP's forward stepwise with glmstar's GLM (the Ch06 lab): no warnings
    # from the null model's empty design
    ISLP = pytest.importorskip('ISLP')
    from ISLP.models import ModelSpec as MS, sklearn_selected
    from ISLP.models.strategy import Stepwise
    Hitters = ISLP.load_data('Hitters').dropna()
    design = MS(Hitters.columns.drop('Salary'), intercept=False).fit(Hitters)
    strategy = Stepwise.first_peak(design, direction='forward', max_terms=3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        selected = sklearn_selected(GLM(), strategy).fit(Hitters, Hitters['Salary'])
    assert len(selected.selected_state_) > 0


@pytest.mark.parametrize('case', ['plain', 'weights', 'offset', 'sparse'])
@pytest.mark.parametrize('name', list(FAMILIES))
def test_score(name, case):
    # score reads the response, weights and offset from y's columns, as fit
    g, s, X_fit, D, offset = _fits(name, case)
    Y = pd.DataFrame({'y': s.model.endog, 'w': s.model.var_weights,
                      'offset': offset if offset is not None else 0.})
    # exactly the deviance at the fit's means, and statsmodels' up to the two
    # fits' convergence
    mu = g.predict(X_fit, offset=offset)
    family, y, w = s.model.family, s.model.endog, s.model.var_weights
    np.testing.assert_allclose(g.score(X_fit, Y), -family.deviance(y, mu, var_weights=w) / 2,
                               rtol=1e-12)
    np.testing.assert_allclose(g.score(X_fit, Y), -s.deviance / 2, rtol=1e-6)
    # sample_weight replaces the weights of weight_id
    np.testing.assert_allclose(g.score(X_fit, Y, sample_weight=np.ones(n)),
                               -family.deviance(y, mu) / 2, rtol=1e-12)


def test_score_cross_validation():
    # cross_val_score on a DataFrame y: each fold's held-out deviance
    from sklearn.model_selection import KFold, cross_val_score
    rng = np.random.default_rng(1)
    eta = 0.3 + X[:, 0] - 0.5 * X[:, 1] + OFFSET
    Y = pd.DataFrame({'y': rng.poisson(np.exp(eta)).astype(float), 'w': W, 'offset': OFFSET})
    est = GLM(family=sm.families.Poisson(), response_id='y', weight_id='w', offset_id='offset')
    scores = cross_val_score(est, X, Y, cv=KFold(5))
    expected = []
    for train, test in KFold(5).split(X):
        fit = GLM(**est.get_params()).fit(X[train], Y.iloc[train])
        mu = np.exp(fit.intercept_ + X[test] @ fit.coef_ + OFFSET[test])
        expected.append(-sm.families.Poisson().deviance(Y['y'].values[test], mu,
                                                        var_weights=W[test]) / 2)
    np.testing.assert_allclose(scores, expected, rtol=1e-8)
