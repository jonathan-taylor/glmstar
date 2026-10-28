# y named by role, built with modelmatrix's Outcome outside fit:
# Y = Outcome(...).fit_transform(D), then fit(X, Y). The estimators read the
# offset, weights and survival data by name; the same fits as by the
# deprecated response_id, offset_id, weight_id and the CoxFamily's ids

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.base import is_classifier
from sklearn.preprocessing import FunctionTransformer

from modelmatrix import Outcome, Surv, Binomial, Multinomial, derived_feature

from glmnet import (GLM, GLMNet, GaussNet, LogNet, FishNet, CoxNet,
                    MultiGaussNet, MultiClassNet, big_glm)
from glmnet.glm import BinomialGLM, GLMFamilySpec
from glmnet.regularized_glm import RegGLM
from glmnet.cox import CoxFamily, CoxLM, CoxNetIRLS

POISSON = GLMFamilySpec(sm.families.Poisson())


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    D = pd.DataFrame({'y': X[:, 0] + rng.standard_normal(n),
                      'y2': X[:, 1] + rng.standard_normal(n),
                      'O': rng.standard_normal(n) / 4,
                      'w': rng.uniform(0.5, 2, n),
                      'exposure': rng.uniform(1, 5, n),
                      'lab': np.where(X[:, 1] + rng.standard_normal(n) > 0, 'yes', 'no'),
                      'cls': rng.choice(['a', 'b', 'c'], n),
                      'start': rng.uniform(0, 0.1, n),
                      'time': rng.exponential(size=n) + 0.2,
                      'status': rng.binomial(1, 0.7, n),
                      'inst': rng.choice(['p', 'q', 'r'], n),
                      'trials': rng.integers(1, 6, n)})
    D['cnt'] = rng.poisson(D['exposure'] * np.exp(0.3 * X[:, 0]))
    D['succ'] = rng.binomial(D['trials'], 0.4)
    D['lab01'] = (D['lab'] == 'yes').astype(float)
    return X, D


def _same_scores(a, b):
    if hasattr(a, 'score_paths'):   # relaxed: one for each gamma
        for a_, b_ in zip(a.score_paths, b.score_paths):
            _same_scores(a_, b_)
        return
    a, b = a.scores, b.scores
    pd.testing.assert_frame_equal(a, b[a.columns], rtol=1e-10)


def _quiet(f, *args, **kwargs):
    # a fit by the deprecated ids (they warn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return f(*args, **kwargs)


# (estimator, Outcome building Y, the same columns by the deprecated ids)
CASES = {
    'glmnet-poisson': (lambda **kw: GLMNet(family=POISSON, **kw),
                       Outcome('cnt', offset='O', weight='w'),
                       dict(response_id='cnt', offset_id='O', weight_id='w')),
    'gaussnet': (GaussNet, Outcome('y', offset='O', weight='w'),
                 dict(response_id='y', offset_id='O', weight_id='w')),
    'fishnet': (FishNet, Outcome('cnt', offset='O', weight='w'),
                dict(response_id='cnt', offset_id='O', weight_id='w')),
    'lognet': (LogNet, Outcome('lab', offset='O', weight='w'),
               dict(response_id='lab', offset_id='O', weight_id='w')),
    'lognet-binomial': (LogNet, Outcome(Binomial('lab'), offset='O'),
                        dict(response_id='lab01', offset_id='O')),
    'lognet-proportion': (LogNet, Outcome(Binomial('succ', trials='trials', output='proportion')),
                          dict(response_id=['trials', 'succ'])),
    'lognet-counts': (LogNet, Outcome(Binomial('succ', trials='trials'), weight='w'),
                      dict(response_id=['trials', 'succ'], weight_id='w')),
    'multigauss': (MultiGaussNet, Outcome(['y', 'y2'], weight='w'),
                   dict(response_id=['y', 'y2'], weight_id='w')),
    'multiclass': (MultiClassNet, Outcome(Multinomial('cls'), weight='w'),
                   dict(response_id='cls', weight_id='w')),
    'coxnet': (CoxNet, Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst'),
               dict(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
                    offset_id='O', weight_id='w')),
    'coxnet-start': (CoxNet, Outcome(Surv('start', 'time', 'status'), weight='w'),
                     dict(family=CoxFamily(event_id='time', status_id='status', start_id='start'),
                          weight_id='w')),
    'coxnet-irls': (CoxNetIRLS, Outcome(Surv('time', 'status'), offset='O'),
                    dict(family=CoxFamily(event_id='time', status_id='status'), offset_id='O')),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_path_matches_ids(name):
    est, outcome, ids = CASES[name]
    X, D = _data()
    Y = outcome.fit_transform(D)
    a = est().fit(X, Y)
    b = _quiet(est(**ids).fit, X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    np.testing.assert_allclose(a.lambda_values_, b.lambda_values_, rtol=1e-10)
    _same_scores(a.score_path(X, Y), _quiet(b.score_path, X, D))
    cv = KFold(4, shuffle=True, random_state=0)
    _same_scores(a.cross_validation_path(X, Y, cv=cv)[1],
                 _quiet(b.cross_validation_path, X, D, cv=cv)[1])


@pytest.mark.parametrize('est, outcome, ids', [
    (lambda **kw: GLM(family=POISSON, **kw), Outcome('cnt', offset='O', weight='w'),
     dict(response_id='cnt', offset_id='O', weight_id='w')),
    (BinomialGLM, Outcome('lab', offset='O'), dict(response_id='lab', offset_id='O')),
    (BinomialGLM, Outcome(Binomial('succ', trials='trials')), dict(response_id=['trials', 'succ'])),
    (lambda **kw: RegGLM(family=POISSON, lambda_val=0.05, **kw), Outcome('cnt', weight='w'),
     dict(response_id='cnt', weight_id='w')),
    (CoxLM, Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst'),
     dict(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
          offset_id='O', weight_id='w')),
])
def test_glm_matches_ids(est, outcome, ids):
    X, D = _data()
    Y = outcome.fit_transform(D)
    a = est().fit(X, Y)
    b = _quiet(est(**ids).fit, X, D)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-10)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=1e-10)
    if not isinstance(a, BinomialGLM):   # ClassifierMixin's score is the accuracy
        np.testing.assert_allclose(a.score(X, Y), _quiet(b.score, X, D), rtol=1e-10)


def test_binomial_cbind():
    # two response columns: R's cbind(successes, failures)
    X, D = _data()
    Y = Outcome(Binomial('succ', trials='trials')).fit_transform(D)
    assert list(Y.columns) == ['succ', 'trials-succ']
    a = LogNet().fit(X, Y)
    b = LogNet().fit(X, Outcome(Binomial('succ', trials='trials',
                                         output='proportion')).fit_transform(D))
    c = LogNet().fit(X, pd.DataFrame({'p': D['succ'] / D['trials'], 'weight': D['trials']}))
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    np.testing.assert_allclose(a.coefs_, c.coefs_, rtol=1e-10)
    with pytest.raises(ValueError, match="cbind"):
        LogNet().fit(X, pd.DataFrame({'s': -np.ones(len(D)), 'f': np.ones(len(D))}))


def test_labels():
    # a column of labels keeps them as the classes; Binomial gives 0/1
    X, D = _data()
    fit = LogNet().fit(X, Outcome('lab').fit_transform(D))
    assert list(fit.classes_) == ['no', 'yes']
    fit = LogNet().fit(X, Outcome(Binomial('lab')).fit_transform(D))
    assert list(fit.classes_) == [0, 1]


def test_multiclass_categories():
    # a categorical column of labels: its levels, in their order
    X, D = _data()
    D['cls'] = pd.Categorical(D['cls'], categories=['c', 'b', 'a'])
    fit = MultiClassNet().fit(X, Outcome(Multinomial('cls')).fit_transform(D))
    assert list(fit.categories_) == ['c', 'b', 'a']
    onehot = np.stack([D['cls'] == c for c in 'cba'], axis=1).astype(float)
    ref = MultiClassNet().fit(X, onehot)
    np.testing.assert_allclose(fit.coefs_, ref.coefs_, rtol=1e-10)


def test_multiresponse_offsets():
    # offset: the same for every response; offset[r]: one for each
    X, D = _data()
    D['O2'] = D['O'] / 2
    a = MultiGaussNet().fit(X, Outcome(['y', 'y2'], offset='O').fit_transform(D))
    b = _quiet(MultiGaussNet(response_id=['y', 'y2'], offset_id=['O', 'O']).fit, X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    Y = Outcome(['y', 'y2'], offset={'y2': 'O2', 'y': 'O'}).fit_transform(D)
    a = MultiGaussNet().fit(X, Y)
    b = _quiet(MultiGaussNet(response_id=['y', 'y2'], offset_id=['O', 'O2']).fit, X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    D['Oa'], D['Ob'], D['Oc'] = D['O'], -D['O'], D['O2']
    outcome = Outcome(Multinomial('cls'), offset={'c': 'Oc', 'a': 'Oa', 'b': 'Ob'})
    Y = outcome.fit_transform(D)
    a = MultiClassNet().fit(X, Y)
    b = _quiet(MultiClassNet(response_id='cls', offset_id=['Oa', 'Ob', 'Oc']).fit, X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    _same_scores(a.score_path(X, Y), _quiet(b.score_path, X, D))
    new = outcome.transform(D.iloc[:5], response=False)
    np.testing.assert_allclose(a.predict(X[:5], offset=new.to_numpy()),
                               b.predict(X[:5], offset=D[['Oa', 'Ob', 'Oc']].iloc[:5].to_numpy()))
    # as with the ids (as in R), a multi-response offset of one column raises
    with pytest.raises(ValueError, match='same shape as the response'):
        _quiet(MultiGaussNet(response_id=['y', 'y2'], offset_id='O').fit, X, D)


def test_feature_offset():
    # R's glm(cnt ~ x, offset=log(exposure), family=poisson)
    X, D = _data()
    log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                                   name='log(exposure)')
    outcome = Outcome('cnt', offset=log_exposure)
    fit = GLM(family=POISSON).fit(X, outcome.fit_transform(D))
    ref = GLM(family=POISSON).fit(X, pd.DataFrame({'cnt': D['cnt'],
                                                   'offset': np.log(D['exposure'])}))
    np.testing.assert_allclose(fit.coef_, ref.coef_, rtol=1e-10)
    new = D.drop(columns='cnt').iloc[:10]
    offset = outcome.transform(new, response=False)['offset']
    np.testing.assert_allclose(fit.predict(X[:10], offset=offset),
                               ref.predict(X[:10], offset=np.log(new['exposure'])), rtol=1e-10)


def test_relaxed_fixed_lambda_survfit():
    X, D = _data()
    Y = Outcome('y', offset='O', weight='w').fit_transform(D)
    a = GaussNet(relax=True).fit(X, Y)
    b = _quiet(GaussNet(relax=True, response_id='y', offset_id='O', weight_id='w').fit, X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    estimator, _ = a.get_fixed_lambda(a.lambda_values_[10])
    np.testing.assert_allclose(estimator.fit(X, Y).coef_,
                               _quiet(b.get_fixed_lambda(a.lambda_values_[10])[0].fit, X, D).coef_,
                               rtol=1e-10)
    Y = Outcome(Surv('time', 'status'), offset='O', strata='inst').fit_transform(D)
    a = CoxNet().fit(X, Y)
    b = _quiet(CoxNet(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
                      offset_id='O').fit, X, D)
    lam = a.lambda_values_[5]
    np.testing.assert_allclose(a.survfit(X, Y, lambda_val=lam).cumulative_hazard,
                               _quiet(b.survfit, X, D, lambda_val=lam).cumulative_hazard, rtol=1e-10)


@pytest.mark.parametrize('name', ['gaussnet', 'lognet', 'coxnet', 'multigauss'])
def test_big_glm(name):
    est, outcome, ids = CASES[name]
    X, D = _data()
    np.testing.assert_allclose(big_glm(est(), X, outcome.fit_transform(D)).coefs_,
                               _quiet(big_glm, est(**ids), X, D).coefs_, rtol=1e-10)


def test_inference():
    from glmnet.inference import glmstar_problem
    X, D = _data()
    Y = Outcome('y', weight='w').fit_transform(D)
    a = GaussNet().fit(X, Y)
    b = _quiet(GaussNet(response_id='y', weight_id='w').fit, X, D)
    lam = a.lambda_values_[10]
    pa = glmstar_problem(a, X, Y, lambda_val=lam)
    pb = _quiet(glmstar_problem, b, X, D, lambda_val=lam)
    np.testing.assert_allclose(pa.beta_hat, pb.beta_hat, rtol=1e-10)


def test_sklearn_and_polars():
    X, D = _data()
    Y = Outcome('cnt', offset='O', weight='w').fit_transform(D)
    ref = GLM(family=POISSON, response_id='cnt', offset_id='O', weight_id='w')
    np.testing.assert_allclose(cross_val_score(GLM(family=POISSON), X, Y, cv=3),
                               _quiet(cross_val_score, ref, X, D, cv=3), rtol=1e-10)
    grid = GridSearchCV(GaussNet(), {'alpha': [0.5, 1]}, cv=3,
                        scoring=lambda e, X, y: -np.mean((e.predict(X)[:, -1] - y['y']) ** 2))
    grid.fit(X, Outcome('y', weight='w').fit_transform(D))
    pl = pytest.importorskip('polars')
    a = GaussNet().fit(X, pl.from_pandas(Outcome('y', weight='w').fit_transform(D)))
    b = GaussNet().fit(X, Outcome('y', weight='w').fit_transform(D))
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)


def test_errors():
    X, D = _data()
    with pytest.raises(ValueError, match='not of its roles'):
        CoxNet().fit(X, D[['time', 'status']].rename(columns={'time': 'stop'}).assign(x=1))
    with pytest.raises(ValueError, match='no columns stop and status'):
        CoxNet().fit(X, D[['y', 'w']])
    with pytest.raises(ValueError, match='weights of each response'):
        MultiGaussNet().fit(X, Outcome(['y', 'y2'], weight={'y': 'w', 'y2': 'w'}).fit_transform(D))
    with pytest.raises(ValueError, match='one or the other'):
        MultiGaussNet().fit(X, pd.DataFrame({'y': D['y'], 'y2': D['y2'], 'offset': D['O'],
                                             'offset[y]': D['O'], 'offset[y2]': D['O']}))


def _future(record):
    return [str(w.message) for w in record if issubclass(w.category, FutureWarning)]


def test_ids_deprecated():
    X, D = _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        GLMNet(family=POISSON, response_id='cnt', offset_id='O', weight_id='w').fit(X, D)
    messages = _future(record)
    assert len(messages) == 1   # once, from the fit called (not the fits within it)
    assert 'response_id, offset_id, weight_id are deprecated' in messages[0]
    with pytest.warns(FutureWarning, match="CoxFamily's event_id"):
        CoxNet(family=CoxFamily(event_id='time', status_id='status')).fit(X, D)
    with pytest.warns(FutureWarning, match="CoxFamily's event_id"):
        CoxNet().fit(X, D[['time', 'status']].rename(columns={'time': 'event'}))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        GaussNet().fit(X, D['y'])
        GaussNet().fit(X, Outcome('y', weight='w').fit_transform(D))
        CoxNet().fit(X, Outcome(Surv('time', 'status')).fit_transform(D))
        MultiGaussNet().fit(X, Outcome(['y', 'y2']).fit_transform(D))
    assert _future(record) == []


# sample_weight, scikit-learn's argument: the column weight of y

def _coef(est):
    return est.coefs_ if hasattr(est, 'coefs_') else est.coef_


@pytest.mark.parametrize('est, outcome', [
    (lambda: GLM(), Outcome('y')),
    (lambda: GLM(family=POISSON), Outcome('cnt', offset='O')),
    (lambda: BinomialGLM(), Outcome('lab01')),
    (lambda: RegGLM(lambda_val=0.05), Outcome('y')),
    (lambda: GLMNet(), Outcome('y')),
    (lambda: GaussNet(), Outcome('y')),
    (lambda: LogNet(), Outcome('lab')),
    (lambda: MultiGaussNet(), Outcome(['y', 'y2'])),
    (lambda: CoxNet(), Outcome(Surv('time', 'status'), strata='inst')),
])
def test_sample_weight_fit(est, outcome):
    X, D = _data()
    Y = outcome.fit_transform(D)
    a = est().fit(X, Y, sample_weight=D['w'].to_numpy())
    b = est().fit(X, Y.assign(weight=D['w'].to_numpy()))
    np.testing.assert_allclose(_coef(a), _coef(b), rtol=1e-10)
    unweighted = est().fit(X, Y)
    # (the weights matter: a path of another length, or other coefficients)
    assert (np.shape(_coef(a)) != np.shape(_coef(unweighted))
            or not np.allclose(_coef(a), _coef(unweighted)))
    with pytest.raises(ValueError, match='given twice'):
        est().fit(X, Y.assign(weight=1.), sample_weight=D['w'].to_numpy())
    if hasattr(a, 'coef_') and not is_classifier(a):   # score (a deviance): the same, given once
        np.testing.assert_allclose(a.score(X, Y, sample_weight=D['w'].to_numpy()),
                                   a.score(X, Y.assign(weight=D['w'].to_numpy())),
                                   rtol=1e-12)
        with pytest.raises(ValueError, match='given twice'):
            a.score(X, Y.assign(weight=1.), sample_weight=D['w'].to_numpy())


def test_sample_weight_array_series():
    # a response alone (an array or a series) and the weights
    X, D = _data()
    w = D['w'].to_numpy()
    ref = GLM().fit(X, Outcome('y', weight='w').fit_transform(D))
    for y in [D['y'], D['y'].to_numpy()]:
        np.testing.assert_allclose(GLM().fit(X, y, sample_weight=w).coef_, ref.coef_, rtol=1e-10)
    ref = LogNet().fit(X, Outcome('lab', weight='w').fit_transform(D))
    np.testing.assert_allclose(LogNet().fit(X, D['lab'], sample_weight=w).coefs_, ref.coefs_,
                               rtol=1e-10)
    with pytest.raises(ValueError, match='120 rows'):
        GLM().fit(X, D['y'], sample_weight=w[:10])


def test_sample_weight_ids():
    # with the deprecated weight_id, sample_weight gives the weights twice
    X, D = _data()
    with pytest.raises(ValueError, match='given twice'):
        _quiet(GLM(response_id='y', weight_id='w').fit, X, D[['y', 'w']],
               sample_weight=D['w'].to_numpy())


def test_sample_weight_cross_validate():
    # cross_validate's params are split with the folds, as the columns of Y
    from sklearn.model_selection import cross_validate
    X, D = _data()
    Y = Outcome('cnt', offset='O').fit_transform(D)
    cv = KFold(3, shuffle=True, random_state=0)
    a = cross_validate(GLM(family=POISSON), X, Y, cv=cv, return_estimator=True,
                       params={'sample_weight': D['w'].to_numpy()})
    b = cross_validate(GLM(family=POISSON), X, Y.assign(weight=D['w'].to_numpy()), cv=cv,
                       return_estimator=True)
    for e, f in zip(a['estimator'], b['estimator']):
        np.testing.assert_allclose(e.coef_, f.coef_, rtol=1e-10)
