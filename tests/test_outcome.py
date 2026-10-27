# outcome=Outcome(...) (modelmatrix): the columns of y named by role, in
# place of response_id, offset_id, weight_id and the CoxFamily's ids

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import FunctionTransformer

from modelmatrix import Outcome, Surv, Binomial, Multinomial, derived_feature

from glmnet import (GLM, GLMNet, GaussNet, LogNet, FishNet, CoxNet,
                    MultiGaussNet, MultiClassNet, Columns)
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
    return X, D


def _same_scores(a, b):
    if hasattr(a, 'score_paths'):   # relaxed: one for each gamma
        for a_, b_ in zip(a.score_paths, b.score_paths):
            _same_scores(a_, b_)
        return
    a, b = a.scores, b.scores
    pd.testing.assert_frame_equal(a, b[a.columns], rtol=1e-10)


# (estimator, Outcome, the same columns by ids)
CASES = {
    'glmnet-poisson': (lambda **kw: GLMNet(family=POISSON, **kw),
                       Outcome('cnt', offset='O', weight='w'),
                       dict(response_id='cnt', offset_id='O', weight_id='w')),
    'gaussnet': (GaussNet, Outcome('y', offset='O', weight='w'),
                 dict(response_id='y', offset_id='O', weight_id='w')),
    'fishnet': (FishNet, Outcome('cnt', offset='O', weight='w'),
                dict(response_id='cnt', offset_id='O', weight_id='w')),
    'lognet': (LogNet, Outcome(Binomial('lab'), offset='O', weight='w'),
               dict(response_id='lab', offset_id='O', weight_id='w')),
    'lognet-proportion': (LogNet, Outcome(Binomial('succ', trials='trials', output='proportion')),
                          dict(response_id=['trials', 'succ'])),
    'lognet-counts': (LogNet, Outcome(Binomial('succ', trials='trials'), weight='w'),
                      dict(response_id=['trials', 'succ'], weight_id='w')),
    'multigauss': (MultiGaussNet, Outcome(Columns(['y', 'y2']), weight='w'),
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
    a = est(outcome=outcome).fit(X, D)
    b = est(**ids).fit(X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    np.testing.assert_allclose(a.lambda_values_, b.lambda_values_, rtol=1e-10)
    _same_scores(a.score_path(X, D), b.score_path(X, D))
    cv = KFold(4, shuffle=True, random_state=0)
    _same_scores(a.cross_validation_path(X, D, cv=cv)[1],
                 b.cross_validation_path(X, D, cv=cv)[1])
    # the outcome given is not modified: a copy is fit
    assert not hasattr(outcome, 'columns_')
    assert hasattr(a.outcome_, 'columns_')


@pytest.mark.parametrize('est, outcome, ids', [
    (lambda **kw: GLM(family=POISSON, **kw), Outcome('cnt', offset='O', weight='w'),
     dict(response_id='cnt', offset_id='O', weight_id='w')),
    (BinomialGLM, Outcome(Binomial('lab'), offset='O'), dict(response_id='lab', offset_id='O')),
    (lambda **kw: RegGLM(family=POISSON, lambda_val=0.05, **kw), Outcome('cnt', weight='w'),
     dict(response_id='cnt', weight_id='w')),
    (CoxLM, Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst'),
     dict(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
          offset_id='O', weight_id='w')),
])
def test_glm_matches_ids(est, outcome, ids):
    X, D = _data()
    a = est(outcome=outcome).fit(X, D)
    b = est(**ids).fit(X, D)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-10)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=1e-10)
    if not isinstance(a, BinomialGLM):   # ClassifierMixin's score is the accuracy
        np.testing.assert_allclose(a.score(X, D), b.score(X, D), rtol=1e-10)


def test_binomial_classes():
    X, D = _data()
    fit = LogNet(outcome=Outcome(Binomial('lab'))).fit(X, D)
    assert list(fit.classes_) == ['no', 'yes']
    # the level of a success, given
    fit = LogNet(outcome=Outcome(Binomial('lab', level='no'))).fit(X, D)
    assert list(fit.classes_) == ['yes', 'no']
    ref = LogNet().fit(X, (D['lab'] == 'no').astype(float))
    np.testing.assert_allclose(fit.coefs_, ref.coefs_, rtol=1e-10)
    np.testing.assert_array_equal(fit.predict(X, prediction_type='class')[:, -1],
                                  np.where(ref.predict(X, prediction_type='class')[:, -1] == 1,
                                           'no', 'yes'))
    fit = BinomialGLM(outcome=Outcome(Binomial('lab'))).fit(X, D)
    assert list(fit.classes_) == ['no', 'yes']
    assert set(fit.predict(X)) <= {'no', 'yes'}


def test_multiclass_categories():
    # the categories are the levels of the Multinomial, of a categorical
    X, D = _data()
    D['cls'] = pd.Categorical(D['cls'], categories=['c', 'b', 'a'])
    fit = MultiClassNet(outcome=Outcome(Multinomial('cls'))).fit(X, D)
    assert list(fit.categories_) == ['c', 'b', 'a']
    onehot = np.stack([D['cls'] == c for c in 'cba'], axis=1).astype(float)
    ref = MultiClassNet().fit(X, onehot)
    np.testing.assert_allclose(fit.coefs_, ref.coefs_, rtol=1e-10)


def test_multiresponse_offset_raises():
    # an offset of a multi-response model has a column for each response
    # (as in R), which an Outcome does not give yet
    X, D = _data()
    with pytest.raises(ValueError, match='same shape as the response'):
        MultiGaussNet(outcome=Outcome(Columns(['y', 'y2']), offset='O')).fit(X, D)
    with pytest.raises(ValueError, match='a column for each class'):
        MultiClassNet(outcome=Outcome(Multinomial('cls'), offset='O')).fit(X, D)


def test_feature_offset():
    # R's glm(cnt ~ x, offset=log(exposure), family=poisson)
    X, D = _data()
    log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                                   name='log(exposure)')
    fit = GLM(family=POISSON, outcome=Outcome('cnt', offset=log_exposure)).fit(X, D)
    D['log_exposure'] = np.log(D['exposure'])
    ref = GLM(family=POISSON, response_id='cnt', offset_id='log_exposure').fit(X, D)
    np.testing.assert_allclose(fit.coef_, ref.coef_, rtol=1e-10)
    # the offset of new data, for predict
    new = D.drop(columns='cnt').iloc[:10]
    offset = fit.outcome_.transform(new, response=False)['offset']
    np.testing.assert_allclose(fit.predict(X[:10], offset=offset),
                               ref.predict(X[:10], offset=new['log_exposure']), rtol=1e-10)


def test_relaxed():
    X, D = _data()
    a = GaussNet(relax=True, outcome=Outcome('y', offset='O', weight='w')).fit(X, D)
    b = GaussNet(relax=True, response_id='y', offset_id='O', weight_id='w').fit(X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)
    cv = KFold(3, shuffle=True, random_state=1)
    _same_scores(a.cross_validation_path(X, D, cv=cv)[1], b.cross_validation_path(X, D, cv=cv)[1])


@pytest.mark.parametrize('name', ['gaussnet', 'lognet', 'coxnet', 'coxnet-irls'])
def test_fixed_lambda(name):
    # the estimator of get_fixed_lambda has the fitted outcome: it is fit
    # to the data as they are, as with the ids
    est, outcome, ids = CASES[name]
    X, D = _data()
    path = est(outcome=outcome).fit(X, D)
    estimator, _ = path.get_fixed_lambda(path.lambda_values_[10])
    assert estimator.outcome is path.outcome_
    # (a RegGLM, LogNet's, takes labels 0/1, as the Binomial codes them)
    R = D.assign(lab=(D['lab'] == 'yes').astype(float))
    ref, _ = est(**ids).fit(X, R).get_fixed_lambda(path.lambda_values_[10])
    np.testing.assert_allclose(estimator.fit(X, D).coef_, ref.fit(X, R).coef_, rtol=1e-10)


def test_survfit():
    X, D = _data()
    a = CoxNet(outcome=Outcome(Surv('time', 'status'), offset='O', strata='inst')).fit(X, D)
    b = CoxNet(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
               offset_id='O').fit(X, D)
    lam = a.lambda_values_[5]
    np.testing.assert_allclose(a.survfit(X, D, lambda_val=lam).cumhaz,
                               b.survfit(X, D, lambda_val=lam).cumhaz, rtol=1e-10)


def test_sklearn():
    X, D = _data()
    est = GLM(family=POISSON, outcome=Outcome('cnt', offset='O', weight='w'))
    ref = GLM(family=POISSON, response_id='cnt', offset_id='O', weight_id='w')
    np.testing.assert_allclose(cross_val_score(est, X, D, cv=3),
                               cross_val_score(ref, X, D, cv=3), rtol=1e-10)
    assert clone(est).get_params()['outcome'] is not est.outcome
    grid = GridSearchCV(GaussNet(outcome=Outcome('y', weight='w')),
                        {'alpha': [0.5, 1]}, cv=3,
                        scoring=lambda e, X, y: -np.mean((e.predict(X)[:, -1] - y['y']) ** 2))
    grid.fit(X, D)
    assert grid.best_params_['alpha'] in [0.5, 1]


def test_prefitted_outcome():
    # an outcome fit already is used as it is, e.g. with the levels of all
    # the data on a subset
    X, D = _data()
    D['cls'] = pd.Categorical(D['cls'])
    outcome = Outcome(Multinomial('cls')).fit(D)
    keep = (D['cls'] != 'c').to_numpy()
    fit = MultiClassNet(outcome=outcome).fit(X[keep], D[keep].assign(
        cls=lambda d: d['cls'].cat.remove_unused_categories()))
    assert list(fit.categories_) == ['a', 'b', 'c']
    assert fit.outcome_ is not outcome


def test_ndarray_and_polars():
    X, D = _data()
    ref = GaussNet(response_id='y', weight_id='w').fit(X, D)
    A = D[['y', 'w']].to_numpy()
    fit = GaussNet(outcome=Outcome(0, weight=1)).fit(X, A)
    np.testing.assert_allclose(fit.coefs_, ref.coefs_, rtol=1e-10)
    pl = pytest.importorskip('polars')
    fit = GaussNet(outcome=Outcome('y', weight='w')).fit(X, pl.from_pandas(D))
    np.testing.assert_allclose(fit.coefs_, ref.coefs_, rtol=1e-10)


def test_errors():
    X, D = _data()
    with pytest.raises(ValueError, match='not both'):
        GaussNet(outcome=Outcome('y'), weight_id='w').fit(X, D)
    with pytest.raises(ValueError, match='not both'):
        CoxNet(outcome=Outcome(Surv('time', 'status')),
               family=CoxFamily(event_id='time')).fit(X, D)
    with pytest.raises(ValueError, match='for a Cox model'):
        GaussNet(outcome=Outcome(Surv('time', 'status'))).fit(X, D)
    with pytest.raises(ValueError, match='Surv response'):
        CoxNet(outcome=Outcome('time')).fit(X, D)
    with pytest.raises(TypeError, match='Outcome'):
        GaussNet(outcome='y').fit(X, D)


@pytest.mark.parametrize('name', ['gaussnet', 'lognet', 'coxnet', 'multigauss'])
def test_big_glm(name):
    from glmnet import big_glm
    est, outcome, ids = CASES[name]
    X, D = _data()
    a = big_glm(est(outcome=outcome), X, D)
    b = big_glm(est(**ids), X, D)
    np.testing.assert_allclose(a.coefs_, b.coefs_, rtol=1e-10)


@pytest.mark.parametrize('name', ['gaussnet', 'lognet', 'fishnet'])
def test_inference(name):
    from glmnet.inference import glmstar_problem, glmstar_inference_problem
    est, outcome, ids = CASES[name]
    X, D = _data()
    a = est(outcome=outcome).fit(X, D)
    b = est(**ids).fit(X, D)
    lam = a.lambda_values_[10]
    pa, pb = glmstar_problem(a, X, D, lambda_val=lam), glmstar_problem(b, X, D, lambda_val=lam)
    np.testing.assert_allclose(pa.beta_hat, pb.beta_hat, rtol=1e-10)
    np.testing.assert_allclose(pa.G_hat, pb.G_hat, rtol=1e-10, atol=1e-12)
    rows = np.arange(X.shape[0]) % 3 != 0
    a = est(outcome=outcome).fit(X[rows], D[rows])
    b = est(**ids).fit(X[rows], D[rows])
    lam = a.lambda_values_[10]
    ia = glmstar_inference_problem(a, X, D, lambda_val=lam, selection_rows=rows)
    ib = glmstar_inference_problem(b, X, D, lambda_val=lam, selection_rows=rows)
    np.testing.assert_allclose(ia.Z_full, ib.Z_full, rtol=1e-10, atol=1e-12)
