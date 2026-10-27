"""
Data frames other than pandas (polars, pyarrow) as inputs: converted to
pandas at the boundary (`glmnet._frames.as_pandas`), so a fit on a polars
frame is the fit on the same pandas frame.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import statsmodels.api as sm
from sklearn.model_selection import KFold

pl = pytest.importorskip('polars')

from sklearn.model_selection import GridSearchCV, cross_val_score

from glmnet import GLM, GLMNet, CoxNet
from glmnet._frames import as_pandas
from glmnet.big_glm import big_glm
from glmnet.cox import CoxFamily, CoxLM
from glmnet.data import make_x
from glmnet.glm import (BinomialGLM, anova, confidence_interval, leverage,
                        prediction_interval)
from glmnet.gam import (GAM, GLMM, ExperimentalWarning, categorical, cubic_regression_spline,
                        linear, random_effect, s, varying)
from glmnet.inference.glmnet_problem import glmstar_problem
from glmnet.regularized_glm import RegGLM

# the estimators of the read-only test, on pandas and on polars frames
from test_readonly import CASES, COX_RESPONSE, RESPONSES, X_DENSE

rng = np.random.default_rng(0)
N = 200
X = pd.DataFrame(rng.standard_normal((N, 4)), columns=['a', 'b', 'c', 'd'])
ETA = X['a'] - 0.5 * X['b']
Y = pd.DataFrame({'y': ETA + rng.standard_normal(N),
                  'w': rng.uniform(0.5, 2, N),
                  'offset': 0.2 * rng.standard_normal(N),
                  'yb': rng.binomial(1, 1 / (1 + np.exp(-ETA))).astype(float)})
T, C = rng.exponential(np.exp(-ETA)), rng.exponential(1, N)
COX = pd.DataFrame({'event': np.minimum(T, C), 'status': (T <= C).astype(float),
                    'strata': rng.integers(0, 3, N)})


def _check(a, b):
    """`a` and `b` the same: arrays, frames, series or tuples of them."""
    if isinstance(a, tuple):
        for u, v in zip(a, b):
            _check(u, v)
    elif isinstance(a, pd.DataFrame):
        # the scorers' columns may come in another order
        pd.testing.assert_frame_equal(a, b[a.columns], rtol=1e-10)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, rtol=1e-10)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


# as_pandas

def test_pass_through():
    A = np.ones((3, 2))
    S = scipy.sparse.csc_matrix(A)
    for obj in [None, A, S, X, X['a'], [1, 2]]:
        assert as_pandas(obj) is obj


def test_numeric_frame():
    df = as_pandas(pl.DataFrame(X.to_dict('list')))
    pd.testing.assert_frame_equal(df, X)
    assert isinstance(df.index, pd.RangeIndex)


def test_dtypes():
    frame = pl.DataFrame({'e': ['lo', 'hi', 'lo'], 'c': ['z', 'x', 'z'],
                          's': ['u', None, 'v'], 'i': [1, None, 3], 'b': [True, False, True]},
                         schema={'e': pl.Enum(['lo', 'hi']), 'c': pl.Categorical,
                                 's': pl.String, 'i': pl.Int64, 'b': pl.Boolean})
    df = as_pandas(frame)
    # an Enum keeps its levels in their order, a Categorical's are sorted
    assert list(df['e'].cat.categories) == ['lo', 'hi']
    assert list(df['c'].cat.categories) == ['x', 'z']
    assert list(df['e']) == ['lo', 'hi', 'lo']
    # (strings: object, or pandas 3's str dtype)
    assert pd.api.types.is_string_dtype(df['s']) and pd.isna(df['s'][1])
    np.testing.assert_array_equal(df['i'], [1, np.nan, 3])
    assert df['b'].dtype == bool


def test_series_and_lazy():
    s_ = as_pandas(pl.Series('z', [1., 2.]))
    pd.testing.assert_series_equal(s_, pd.Series([1., 2.], name='z'))
    lazy = pl.DataFrame(X.to_dict('list')).lazy()
    pd.testing.assert_frame_equal(as_pandas(lazy), X)


# fits on polars frames equal fits on pandas frames

def _pl(df):
    return pl.DataFrame(df.to_dict('list'))


@pytest.mark.parametrize('family, response', [(sm.families.Gaussian(), 'y'),
                                              (sm.families.Binomial(), 'yb')])
def test_glm(family, response):
    kw = dict(family=family, summarize=True, response_id=response, weight_id='w',
              offset_id='offset')
    fits = [GLM(**kw).fit(x, y) for x, y in [(X, Y), (_pl(X), _pl(Y))]]
    _check(fits[0].coef_, fits[1].coef_)
    _check(fits[0].summary_, fits[1].summary_)
    _check(fits[0].predict(X, offset=Y['offset']),
           fits[1].predict(_pl(X), offset=Y['offset'].to_numpy()))
    _check(confidence_interval(fits[0], X), confidence_interval(fits[1], _pl(X)))
    _check(leverage(fits[0], X), leverage(fits[1], _pl(X)))
    assert fits[1].feature_names_in_ == list(X.columns)


def test_anova():
    tables = []
    for x, y in [(X, Y), (_pl(X), _pl(Y))]:
        small = GLM(summarize=True, response_id='y').fit(x[['a', 'b']], y)
        big = GLM(summarize=True, response_id='y').fit(x, y)
        tables.append(anova(small, big))
    _check(*tables)


def test_glmnet_cv():
    fits, cvs = [], []
    for x, y in [(X, Y), (_pl(X), _pl(Y))]:
        fit = GLMNet(response_id='y', weight_id='w').fit(x, y)
        cvs.append(fit.cross_validation_path(x, y, cv=KFold(5)))
        fits.append(fit)
    _check(fits[0].coefs_, fits[1].coefs_)
    _check(fits[0].predict(X), fits[1].predict(_pl(X)))
    _check(cvs[0][0], cvs[1][0])
    _check(cvs[0][1].scores, cvs[1][1].scores)


def test_coxnet_strata():
    fam = dict(family=CoxFamily(tie_breaking='efron', strata_id='strata'))
    fits = [CoxNet(**fam).fit(x, y) for x, y in [(X, COX), (_pl(X), _pl(COX))]]
    _check(fits[0].coefs_, fits[1].coefs_)
    _check(fits[0].predict(X), fits[1].predict(_pl(X)))


def _gam_data():
    rng = np.random.default_rng(1)
    data = X.copy()
    data['g'] = pd.Categorical(rng.choice(['lo', 'mid', 'hi'], N),
                               categories=['lo', 'mid', 'hi'])
    data['y'] = (np.sin(2 * data['a']) + data['g'].cat.codes * 0.5
                 + 0.3 * rng.standard_normal(N))
    frame = _pl(data.assign(g=data['g'].astype(str))).with_columns(
        pl.col('g').cast(pl.Enum(['lo', 'mid', 'hi'])))
    return data, frame


def test_gam():
    data, frame = _gam_data()
    terms = [s('a', df=4), linear('b'), categorical('g'),
             varying(cubic_regression_spline('c', k=5), by='g', lam=1.)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        fits = [GAM(terms).fit(d, d['y'] if isinstance(d, pd.DataFrame) else d['y'])
                for d in [data, frame]]
    _check(fits[0].edf_, fits[1].edf_)
    _check(fits[0].predict(data), fits[1].predict(frame))
    _check(fits[0].partial_effect(2, data), fits[1].partial_effect(2, frame))
    assert list(fits[1].terms_[2].levels_) == ['lo', 'mid', 'hi']


def test_make_x():
    data, frame = _gam_data()
    _check(make_x(data[['a', 'g']]), make_x(frame.select('a', 'g')))


def test_pyarrow_table():
    pa = pytest.importorskip('pyarrow')
    kw = dict(summarize=True, response_id='y', weight_id='w')
    a = GLM(**kw).fit(X, Y)
    b = GLM(**kw).fit(pa.Table.from_pandas(X), pa.Table.from_pandas(Y))
    _check(a.coef_, b.coef_)
    _check(a.summary_, b.summary_)


# every estimator of test_readonly.py

def _frame_response(kind):
    if kind == 'cox':
        return COX_RESPONSE
    r = RESPONSES[kind]
    return pd.DataFrame(r.reshape((r.shape[0], -1)),
                        columns=['y'] if r.ndim == 1 else [f'y{j}' for j in range(r.shape[1])])


@pytest.mark.parametrize('name', list(CASES))
def test_estimators(name):
    make, kind = CASES[name]
    x = pd.DataFrame(X_DENSE, columns=[f'x{j}' for j in range(X_DENSE.shape[1])])
    y = _frame_response(kind)
    a, b = make().fit(x, y), make().fit(_pl(x), _pl(y))
    coef = lambda m: np.asarray(m.coefs_ if hasattr(m, 'coefs_') else m.coef_)
    _check(coef(a), coef(b))
    _check(a.predict(x), b.predict(_pl(x)))


# the other entry points

def test_glm_entry_points():
    kw = dict(summarize=True, response_id='y', weight_id='w')
    a, b = GLM(**kw).fit(X, Y), GLM(**kw).fit(_pl(X), _pl(Y))
    _check(a.predict_se(X), b.predict_se(_pl(X)))
    _check(prediction_interval(a, X), prediction_interval(b, _pl(X)))
    _check(a.score(X, Y), b.score(_pl(X), _pl(Y)))
    c = BinomialGLM(response_id='yb').fit(X, Y)
    d = BinomialGLM(response_id='yb').fit(_pl(X), _pl(Y))
    _check(c.predict_proba(X), d.predict_proba(_pl(X)))
    _check(c.predict(X), d.predict(_pl(X)))


def test_big_glm_and_problem():
    est = GLMNet(response_id='y', weight_id='w')
    _check(big_glm(est, X, Y).coefs_, big_glm(est, _pl(X), _pl(Y)).coefs_)
    fit = est.fit(X, Y)
    lam = fit.lambda_values_[10]
    a = glmstar_problem(fit, X, Y, lambda_val=lam)
    b = glmstar_problem(fit, _pl(X), _pl(Y), lambda_val=lam)
    _check(a.beta_hat, b.beta_hat)
    _check(a.G_hat, b.G_hat)


def test_cox_string_strata():
    # a mixed frame: numeric columns come from polars as read-only views
    event = COX.assign(strata=COX['strata'].map({0: 'a', 1: 'b', 2: 'c'}))
    fam = lambda: CoxFamily(tie_breaking='efron', strata_id='strata')
    frame = _pl(event)
    assert not as_pandas(frame)['event'].to_numpy().flags.writeable
    a, b = CoxLM(family=fam()).fit(X, event), CoxLM(family=fam()).fit(_pl(X), frame)
    _check(a.coef_, b.coef_)
    _check(a.score(X, event), b.score(_pl(X), frame))


def test_gam_cox_and_score():
    data = pd.concat([X, COX], axis=1)
    fam = lambda: CoxFamily(event_id='event', status_id='status')
    terms = [s('a', df=4), linear('b')]
    a = GAM(terms, family=fam()).fit(data, data)
    b = GAM(terms, family=fam()).fit(_pl(data), _pl(data))
    _check(a.edf_, b.edf_)
    _check(a.predict(data), b.predict(_pl(data)))
    data, frame = _gam_data()
    a, b = GAM([s('a', df=4)]).fit(data, data['y']), GAM([s('a', df=4)]).fit(frame, frame['y'])
    _check(a.score(data, data['y']), b.score(frame, frame['y']))


def test_glmm():
    pytest.importorskip('jax')
    data, frame = _gam_data()
    terms = [s('a'), random_effect('g')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        a, b = GLMM(terms).fit(data, data['y']), GLMM(terms).fit(frame, frame['y'])
    _check(a.smoothing_.lam, b.smoothing_.lam)
    _check(a.predict(data), b.predict(frame))


# edge cases

def test_nulls_raise():
    frame = _pl(X).with_columns(pl.when(pl.col('a') > 1).then(None).otherwise(pl.col('a'))
                                .alias('a'))
    with pytest.raises(ValueError, match='NaN'):
        GLM(response_id='y').fit(frame, _pl(Y))


def test_gam_polars_categorical():
    # a Categorical's levels are sorted, as a pandas column of strings
    data, _ = _gam_data()
    data = data.assign(g=data['g'].astype(str))
    frame = _pl(data).with_columns(pl.col('g').cast(pl.Categorical))
    terms = [s('a', df=4), categorical('g')]
    a, b = GAM(terms).fit(data, data['y']), GAM(terms).fit(frame, frame['y'])
    assert list(a.terms_[1].levels_) == list(b.terms_[1].levels_)
    _check(a.predict(data), b.predict(frame))


def test_columns_reordered():
    # predict sees the columns by name, as for pandas
    kw = dict(summarize=True, response_id='y')
    a, b = GLM(**kw).fit(X, Y), GLM(**kw).fit(_pl(X), _pl(Y))
    for fit, x in [(a, X[['b', 'a', 'c', 'd']]), (b, _pl(X).select('b', 'a', 'c', 'd'))]:
        with pytest.raises(ValueError, match='feature names'):
            fit.predict(x)


def test_lazy_fit():
    kw = dict(summarize=True, response_id='y')
    a, b = GLM(**kw).fit(X, Y), GLM(**kw).fit(_pl(X).lazy(), _pl(Y).lazy())
    _check(a.coef_, b.coef_)


# sklearn slices the polars frames itself

def test_sklearn():
    # the response, weights and offset as columns of y, in each fold
    kw = dict(response_id='y', weight_id='w', offset_id='offset')
    a = cross_val_score(GLM(**kw), X, Y, cv=KFold(5))
    b = cross_val_score(GLM(**kw), _pl(X), _pl(Y), cv=KFold(5))
    assert np.all(np.isfinite(a))
    _check(a, b)
    grid = {'lambda_val': [0.01, 0.1]}
    a = GridSearchCV(RegGLM(**kw), grid, cv=KFold(5)).fit(X, Y)
    b = GridSearchCV(RegGLM(**kw), grid, cv=KFold(5)).fit(_pl(X), _pl(Y))
    assert np.all(np.isfinite(a.cv_results_['mean_test_score']))
    _check(a.cv_results_['mean_test_score'], b.cv_results_['mean_test_score'])
    _check(a.best_estimator_.coef_, b.best_estimator_.coef_)
