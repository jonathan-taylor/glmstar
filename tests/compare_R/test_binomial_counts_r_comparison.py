"""
Binomial responses other than 0/1 labels against R: counts of successes out of
a number of trials, and proportions with the number of trials as weights.

glmstar's forms
---------------
- `LogNet`: a two-column response (trials, successes) -- note the order: R's
  `glmnet` takes `cbind(failures, successes)` and R's `glm` takes
  `cbind(successes, failures)` -- or proportions in [0, 1] with `weight_id`
  the number of trials;
- `GLMNet(family=Binomial())` and `GLM(family=Binomial())`: proportions with
  weights only (a two-column response is not supported);
- `BinomialGLM` and `BinomialRegGLM`: (trials, successes) only (proportions
  are taken as labels, and rejected as more than two classes).

The tests compare the lasso path (lambda values, coefficients, intercepts,
dev.ratio and predictions), with and without an offset, with R's `glmnet`
given the equivalent input; cross-validation with `cv.glmnet`; the equivalence
of counts, proportions + weights and the 0/1 data with one row per trial; and
the unpenalized fit with R's `glm(cbind(s, f) ~ X, binomial)`.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit
from sklearn.model_selection import PredefinedSplit

from glmnet import GLM, GLMNet, LogNet
from glmnet.glm import BinomialGLM, GLMControl
from glmnet.glmnet import GLMNetControl
from glmnet.paths.fastnet import FastNetControl

N, P, N_NEW = 120, 8, 15
CONTROL = FastNetControl(thresh=1e-14)
TIGHT_IRLS = GLMNetControl(thresh=1e-14, epsnr=1e-12, mxitnr=100)
TIGHT_R = 'control=list(thresh=1e-14, epsnr=1e-12, mxitnr=100)'
TIGHT_GLM = GLMControl(epsnr=1e-12, mxitnr=100)
TOL = 1e-8


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, P))
    o = rng.normal(0, 0.3, N)
    t = rng.integers(1, 12, N).astype(float)
    s = rng.binomial(t.astype(int), expit(0.2 + X[:, 0] - 0.5 * X[:, 1] + o)).astype(float)
    D = pd.DataFrame({'t': t, 's': s, 'prop': s / t, 'o': o})
    Xnew = rng.standard_normal((N_NEW, P))
    onew = rng.normal(0, 0.3, N_NEW)
    foldid = rng.permutation(np.arange(N) % 5 + 1)
    return X, D, Xnew, onew, foldid


def _skip(Rinfo):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')


def _assign(Rinfo, **arrays):
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        for k, v in arrays.items():
            rpy.r.assign(k, np.asarray(v, float))
    for k, v in arrays.items():
        if np.asarray(v).ndim == 1:
            rpy.r(f'{k} <- as.vector({k})')


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr), float)


def _to_R(Rinfo, data):
    X, D, Xnew, onew, foldid = data
    _assign(Rinfo, X=X, s=D['s'], t=D['t'], o=D['o'], Xnew=Xnew, onew=onew, foldid=foldid)
    Rinfo['rpy'].r('suppressMessages(library(glmnet)); foldid <- as.integer(foldid); f <- t - s')


def _lognet(form, with_offset, **kw):
    if with_offset:
        kw['offset_id'] = 'o'
    if form == 'counts':
        return LogNet(response_id=['t', 's'], control=CONTROL, **kw)
    return LogNet(response_id='prop', weight_id='t', control=CONTROL, **kw)


def _check_path(Rinfo, fit, tol=TOL):
    # against the R fit G
    R_lambda = _get(Rinfo, 'G$lambda')
    assert fit.lambda_values_.shape == R_lambda.shape
    np.testing.assert_allclose(fit.lambda_values_, R_lambda, rtol=1e-10)
    np.testing.assert_allclose(fit.coefs_, _get(Rinfo, 'as.matrix(G$beta)').T, rtol=tol, atol=tol)
    np.testing.assert_allclose(fit.intercepts_, _get(Rinfo, 'G$a0'), rtol=tol, atol=tol)
    np.testing.assert_allclose(fit.summary_['Fraction Deviance Explained'], _get(Rinfo, 'G$dev.ratio'),
                               rtol=0, atol=tol)


def _check_predict(Rinfo, fit, Xnew, onew, with_offset, tol=TOL):
    off, Roff = (onew, ', newoffset=onew') if with_offset else (None, '')
    for kind in ['link', 'response']:
        theirs = _get(Rinfo, f'predict(G, Xnew, type="{kind}"{Roff})')
        # glmstar pads its predictions to nlambda columns
        ours = np.asarray(fit.predict(Xnew, prediction_type=kind, offset=off)).reshape(N_NEW, -1)
        ours = ours[:, :theirs.shape[1]]
        np.testing.assert_allclose(ours, theirs, rtol=tol, atol=tol)


# the lasso path: LogNet

@pytest.mark.parametrize('with_offset', [False, True])
@pytest.mark.parametrize('form', ['counts', 'proportion'])
def test_lognet_vs_R_counts(Rinfo, data, form, with_offset):
    # R glmnet with the two-column matrix of counts cbind(failures, successes)
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    L = _lognet(form, with_offset).fit(X, D)
    _to_R(Rinfo, data)
    Roff = ', offset=o' if with_offset else ''
    Rinfo['rpy'].r(f'G <- glmnet(X, cbind(f, s), family="binomial", control=list(thresh=1e-14){Roff})')
    _check_path(Rinfo, L)
    _check_predict(Rinfo, L, Xnew, onew, with_offset)
    assert list(L.classes_) == [0, 1]


@pytest.mark.parametrize('with_offset', [False, True])
def test_R_counts_vs_proportions(Rinfo, data, with_offset):
    # in R itself: counts, the matrix of proportions with weights, and the
    # glmnet.path fit of proportions with weights give the same path
    _skip(Rinfo)
    _to_R(Rinfo, data)
    rpy = Rinfo['rpy']
    Roff = ', offset=o' if with_offset else ''
    rpy.r(f'''G <- glmnet(X, cbind(f, s), family="binomial", control=list(thresh=1e-14){Roff})
Gp <- glmnet(X, cbind(f, s) / t, weights=t, family="binomial", control=list(thresh=1e-14){Roff})
Gf <- glmnet(X, s / t, weights=t, family=binomial(), lambda=G$lambda, {TIGHT_R}{Roff})''')
    for other in ['Gp', 'Gf']:
        np.testing.assert_allclose(_get(Rinfo, f'as.matrix({other}$beta)'), _get(Rinfo, 'as.matrix(G$beta)'),
                                   rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(_get(Rinfo, f'{other}$a0'), _get(Rinfo, 'G$a0'), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(_get(Rinfo, f'{other}$dev.ratio'), _get(Rinfo, 'G$dev.ratio'), atol=1e-8)
    np.testing.assert_allclose(_get(Rinfo, 'Gp$lambda'), _get(Rinfo, 'G$lambda'), rtol=1e-12)


def test_lognet_cv_counts_offset_vs_R(Rinfo, data):
    # cv.glmnet with foldid, counts and an offset
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    L = _lognet('counts', True).fit(X, D)
    L.cross_validation_path(X, D, cv=PredefinedSplit(foldid))
    scores = L.score_path_.scores
    _to_R(Rinfo, data)
    Rinfo['rpy'].r('CV <- cv.glmnet(X, cbind(f, s), family="binomial", offset=o, foldid=foldid, '
                   'type.measure="deviance", control=list(thresh=1e-14))')
    np.testing.assert_allclose(L.lambda_values_, _get(Rinfo, 'CV$lambda'), rtol=1e-10)
    np.testing.assert_allclose(scores['Binomial Deviance'].values, _get(Rinfo, 'CV$cvm'), rtol=TOL, atol=1e-12)
    np.testing.assert_allclose(scores['SD(Binomial Deviance)'].values, _get(Rinfo, 'CV$cvsd'),
                               rtol=TOL, atol=1e-12)
    assert np.isclose(L.score_path_.index_best['Binomial Deviance'], _get(Rinfo, 'CV$lambda.min')[0])
    assert np.isclose(L.score_path_.index_1se['Binomial Deviance'], _get(Rinfo, 'CV$lambda.1se')[0])


# the lasso path: GLMNet (IRLS) with the statsmodels Binomial family

@pytest.mark.parametrize('with_offset', [False, True])
def test_glmnet_proportions_vs_R(Rinfo, data, with_offset):
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    kw = dict(family=sm.families.Binomial(), response_id='prop', weight_id='t')
    if with_offset:
        kw['offset_id'] = 'o'
    _to_R(Rinfo, data)
    rpy = Rinfo['rpy']
    Roff = ', offset=o' if with_offset else ''
    # the default path: R's lambda values (from the counts) and path length
    rpy.r(f'G0 <- glmnet(X, cbind(f, s), family="binomial"{Roff})')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0 = GLMNet(**kw).fit(X, D)
    R_lambda0 = _get(Rinfo, 'G0$lambda')
    assert G0.lambda_values_.shape == R_lambda0.shape
    np.testing.assert_allclose(G0.lambda_values_, R_lambda0, rtol=1e-6)

    # a tight fit at those lambda values, against R's IRLS (glmnet.path) and lognet
    rpy.r(f'G <- glmnet(X, s / t, weights=t, family=binomial(), lambda=G0$lambda, {TIGHT_R}{Roff})')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = GLMNet(lambda_values=R_lambda0, control=TIGHT_IRLS, **kw).fit(X, D)
    # the IRLS paths agree to ~1e-7 at the end of the path, as in test_irls_r_parity.py
    _check_path(Rinfo, G, tol=1e-6)
    _check_predict(Rinfo, G, Xnew, onew, with_offset, tol=1e-6)
    rpy.r(f'G <- glmnet(X, cbind(f, s), family="binomial", lambda=G0$lambda, control=list(thresh=1e-14){Roff})')
    np.testing.assert_allclose(G.coefs_, _get(Rinfo, 'as.matrix(G$beta)').T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(G.intercepts_, _get(Rinfo, 'G$a0'), rtol=1e-6, atol=1e-6)


# the equivalence of counts, proportions + weights and 0/1 data

def _expand(X, D):
    # one row per trial, successes first
    t, s = D['t'].values.astype(int), D['s'].values.astype(int)
    idx = np.repeat(np.arange(len(t)), t)
    y = np.concatenate([np.r_[np.ones(si), np.zeros(ti - si)] for ti, si in zip(t, s)])
    return X[idx], pd.DataFrame({'y': y, 'o': D['o'].values[idx]})


@pytest.mark.parametrize('with_offset', [False, True])
def test_counts_proportions_expanded_equivalent(Rinfo, data, with_offset):
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    Lc = _lognet('counts', with_offset).fit(X, D)
    Lp = _lognet('proportion', with_offset).fit(X, D)
    Xe, De = _expand(X, D)
    kw = dict(offset_id='o') if with_offset else {}
    Le = LogNet(response_id='y', control=CONTROL, **kw).fit(Xe, De)

    # counts and proportions: the same problem
    np.testing.assert_allclose(Lp.lambda_values_, Lc.lambda_values_, rtol=1e-12)
    np.testing.assert_allclose(Lp.coefs_, Lc.coefs_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Lp.intercepts_, Lc.intercepts_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Lp.summary_['Fraction Deviance Explained'],
                               Lc.summary_['Fraction Deviance Explained'], atol=1e-12)

    # the expanded 0/1 data: the same coefficients, though not the same
    # dev.ratio -- both deviances gain -2 times the saturated log-likelihood
    # of the grouped data, which is 0 for 0/1 data
    k = min(len(Le.lambda_values_), len(Lc.lambda_values_))
    np.testing.assert_allclose(Le.lambda_values_[:k], Lc.lambda_values_[:k], rtol=1e-10)
    np.testing.assert_allclose(Le.coefs_[:k], Lc.coefs_[:k], rtol=TOL, atol=TOL)
    np.testing.assert_allclose(Le.intercepts_[:k], Lc.intercepts_[:k], rtol=TOL, atol=TOL)
    dev_c = 1 - Lc.summary_['Fraction Deviance Explained'].values[:k]
    dev_e = 1 - Le.summary_['Fraction Deviance Explained'].values[:k]
    assert not np.allclose(dev_c, dev_e)

    # and R: the expanded data against the counts
    _to_R(Rinfo, data)
    _assign(Rinfo, Xe=Xe, ye=De['y'], oe=De['o'])
    Roff, Roffe = (', offset=o', ', offset=oe') if with_offset else ('', '')
    Rinfo['rpy'].r(f'''G <- glmnet(X, cbind(f, s), family="binomial", control=list(thresh=1e-14){Roff})
Ge <- glmnet(Xe, ye, family="binomial", lambda=G$lambda, control=list(thresh=1e-14){Roffe})''')
    np.testing.assert_allclose(Le.coefs_[:k], _get(Rinfo, 'as.matrix(Ge$beta)').T[:k], rtol=TOL, atol=TOL)
    np.testing.assert_allclose(_get(Rinfo, 'as.matrix(Ge$beta)'), _get(Rinfo, 'as.matrix(G$beta)'),
                               rtol=TOL, atol=TOL)


def test_glmnet_proportions_vs_lognet_counts(data):
    # GLMNet's IRLS with proportions + weights and LogNet with counts
    X, D, Xnew, onew, foldid = data
    Lc = _lognet('counts', True).fit(X, D)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = GLMNet(family=sm.families.Binomial(), response_id='prop', weight_id='t', offset_id='o',
                   lambda_values=Lc.lambda_values_, control=TIGHT_IRLS).fit(X, D)
    np.testing.assert_allclose(G.coefs_, Lc.coefs_, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(G.intercepts_, Lc.intercepts_, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(G.summary_['Fraction Deviance Explained'],
                               Lc.summary_['Fraction Deviance Explained'], atol=1e-8)


# the unpenalized fit: R's glm(cbind(s, f) ~ X, binomial)

def _glm(form, with_offset):
    kw = dict(summarize=True, control=TIGHT_GLM)
    if with_offset:
        kw['offset_id'] = 'o'
    if form == 'GLM_proportion':
        return GLM(family=sm.families.Binomial(), response_id='prop', weight_id='t', **kw)
    return BinomialGLM(response_id=['t', 's'], **kw)


def _glm_R(Rinfo, data, with_offset):
    _to_R(Rinfo, data)
    Roff = ' + offset(o)' if with_offset else ''
    Rinfo['rpy'].r(f'''F <- glm(cbind(s, f) ~ X{Roff}, family=binomial,
                          control=glm.control(epsilon=1e-14, maxit=100))
N <- data.frame(X=I(Xnew), o=onew)
Pl <- predict(F, newdata=N, type="link", se.fit=TRUE)''')


@pytest.mark.parametrize('with_offset', [False, True])
@pytest.mark.parametrize('form', ['GLM_proportion', 'BinomialGLM_counts'])
def test_glm_vs_R(Rinfo, data, form, with_offset):
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    fit = _glm(form, with_offset).fit(X, D)
    _glm_R(Rinfo, data, with_offset)
    get = lambda expr: _get(Rinfo, expr)

    np.testing.assert_allclose(fit.params_.values, get('coef(F)'), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fit.summary_['std err'].values, get('sqrt(diag(vcov(F)))'), rtol=1e-8)
    np.testing.assert_allclose(fit.summary_['P>|z|'].values, get('summary(F)$coefficients[, 4]'), rtol=1e-7)
    np.testing.assert_allclose(fit.deviance_, get('deviance(F)')[0], rtol=1e-10)
    np.testing.assert_allclose(fit.null_deviance_, get('F$null.deviance')[0], rtol=1e-10)

    off = onew if with_offset else None
    np.testing.assert_allclose(fit.predict(Xnew, prediction_type='link', offset=off), get('Pl$fit'), rtol=1e-9)
    np.testing.assert_allclose(fit.predict(Xnew, prediction_type='response', offset=off),
                               get('predict(F, newdata=N, type="response")'), rtol=1e-9)
    np.testing.assert_allclose(fit.predict_se(Xnew), get('Pl$se.fit'), rtol=1e-8)


@pytest.mark.xfail(reason="aic_ / loglike_ of grouped binomial data omit R's log(choose(t, s)) and add "
                          "-t * (lgamma(1 + y) + lgamma(2 - y)): statsmodels' Binomial.loglike "
                          "with n=1 and var_weights=t", strict=True)
@pytest.mark.parametrize('form', ['GLM_proportion', 'BinomialGLM_counts'])
def test_glm_aic_vs_R(Rinfo, data, form):
    _skip(Rinfo)
    X, D, Xnew, onew, foldid = data
    fit = _glm(form, False).fit(X, D)
    _glm_R(Rinfo, data, False)
    np.testing.assert_allclose(fit.loglike_, _get(Rinfo, 'logLik(F)')[0], rtol=1e-10)
    np.testing.assert_allclose(fit.aic_, _get(Rinfo, 'AIC(F)')[0], rtol=1e-10)
