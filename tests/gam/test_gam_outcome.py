# outcome=Outcome(...) (modelmatrix) for GAM and GLMM: the same fits as by
# response_id, offset_id, weight_id and the CoxFamily's ids

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer

pytest.importorskip('scatter_smooth')

from modelmatrix import Outcome, Surv, Binomial, derived_feature

from glmnet.cox import CoxFamily
from glmnet.family import GLMFamilySpec
from glmnet.gam import GAM, GLMM, s, linear, categorical, random_effect
from glmnet.gam.reml import ExperimentalWarning

POISSON = sm.families.Poisson()
BINOMIAL = sm.families.Binomial()


def _data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({'x': rng.uniform(0, 1, n),
                      'z': rng.standard_normal(n),
                      'c': rng.choice(list('abc'), n),
                      'g': rng.choice([f'g{i}' for i in range(8)], n),
                      'w': rng.uniform(0.5, 2, n),
                      'O': rng.standard_normal(n) / 4,
                      'exposure': rng.uniform(1, 5, n),
                      'trials': rng.integers(1, 8, n),
                      'inst': rng.choice(['p', 'q'], n)})
    eta = np.sin(2 * np.pi * D['x']) + 0.5 * D['z']
    D['y'] = eta + rng.standard_normal(n)
    D['cnt'] = rng.poisson(D['exposure'] * np.exp(0.5 * eta))
    D['lab'] = np.where(eta + rng.logistic(size=n) > 0, 'yes', 'no')
    D['succ'] = rng.binomial(D['trials'], 1 / (1 + np.exp(-eta)))
    D['time'] = rng.exponential(1 / np.exp(eta)) + 0.01
    D['status'] = rng.binomial(1, 0.7, n)
    return D


TERMS = [s('x', df=4), linear('z')]

# (family, Outcome, the same columns by ids, of the data recoded by _ref_data)
CASES = {
    'gaussian': (None, Outcome('y', offset='O', weight='w'),
                 dict(response_id='y', offset_id='O', weight_id='w')),
    'poisson': (POISSON, Outcome('cnt', offset='O', weight='w'),
                dict(response_id='cnt', offset_id='O', weight_id='w')),
    'binomial': (BINOMIAL, Outcome(Binomial('lab'), weight='w'),
                 dict(response_id='lab01', weight_id='w')),
    'binomial-trials': (BINOMIAL, Outcome(Binomial('succ', trials='trials')),
                        dict(response_id='prop', weight_id='trials')),
    'binomial-trials-weights': (BINOMIAL, Outcome(Binomial('succ', trials='trials',
                                                           output='proportion'), weight='w'),
                                dict(response_id='prop', weight_id='tw')),
    'cox': (CoxFamily(), Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst'),
            dict(family=CoxFamily(event_id='time', status_id='status', strata_id='inst'),
                 offset_id='O', weight_id='w')),
}


def _ref_data(D):
    # the responses recoded, as the ids take them
    return D.assign(lab01=(D['lab'] == 'yes').astype(float),
                    prop=D['succ'] / D['trials'],
                    tw=D['trials'] * D['w'])


def _gam(est, case_family, terms=TERMS, **kw):
    # the family of the case, unless the ids give one (a CoxFamily's)
    if case_family is not None:
        kw.setdefault('family', case_family)
    return est(terms, **kw)


@pytest.mark.parametrize('name', sorted(CASES))
def test_gam_matches_ids(name):
    family, outcome, ids = CASES[name]
    D = _data()
    R = _ref_data(D)
    a = _gam(GAM, family, outcome=outcome).fit(D, D)
    b = _gam(GAM, family, **ids).fit(R, R)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.edf_, b.edf_, rtol=1e-9)
    np.testing.assert_allclose(a.score(D, D), b.score(R, R), rtol=1e-9)
    assert not hasattr(outcome, 'columns_')   # a copy is fit


@pytest.mark.parametrize('name', ['gaussian', 'poisson', 'binomial-trials', 'cox'])
def test_glmm_matches_ids(name):
    pytest.importorskip('jax')
    family, outcome, ids = CASES[name]
    D = _data()
    R = _ref_data(D)
    terms = [s('x'), random_effect('g')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        a = _gam(GLMM, family, terms, outcome=outcome).fit(D, D)
        b = _gam(GLMM, family, terms, **ids).fit(R, R)
    np.testing.assert_allclose(a.smoothing_.lam, b.smoothing_.lam, rtol=1e-6)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-6, atol=1e-9)


def test_feature_offset():
    # mgcv's gam(cnt ~ s(x) + z + offset(log(exposure)), family=poisson)
    D = _data()
    log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                                   name='log(exposure)')
    a = GAM(TERMS, family=POISSON, outcome=Outcome('cnt', offset=log_exposure)).fit(D, D)
    R = D.assign(log_exposure=np.log(D['exposure']))
    b = GAM(TERMS, family=POISSON, response_id='cnt', offset_id='log_exposure').fit(R, R)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-9)
    offset = a.outcome_.transform(D.iloc[:10], response=False)['offset']
    np.testing.assert_allclose(a.predict(D.iloc[:10], offset=offset),
                               b.predict(R.iloc[:10], offset=R['log_exposure'].iloc[:10]),
                               rtol=1e-9)
    assert r'\text{log(exposure)}_i' in a.model_latex()


def test_latex():
    D = _data()
    gam = GAM(TERMS, family=BINOMIAL,
              outcome=Outcome(Binomial('succ', trials='trials'), offset='O', weight='w'))
    tex = gam.model_latex()   # unfitted, as Jupyter shows it
    assert r'\text{O}_i' in tex
    assert r'w_i &= \text{trials}_i \cdot \text{w}_i' in tex
    ids = GAM(TERMS, offset_id='O', weight_id='w').model_latex()
    assert r'\text{O}_i' in ids and r'w_i &= \text{w}_i' in ids
    gam.fit(D, D)
    assert gam.model_latex() == tex


def test_sklearn():
    D = _data()
    cv = KFold(4, shuffle=True, random_state=0)
    a = cross_val_score(GAM(TERMS, family=POISSON, outcome=Outcome('cnt', offset='O')), D, D, cv=cv)
    b = cross_val_score(GAM(TERMS, family=POISSON, response_id='cnt', offset_id='O'), D, D, cv=cv)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_errors():
    D = _data()
    with pytest.raises(ValueError, match='not both'):
        GAM(TERMS, outcome=Outcome('y'), weight_id='w').fit(D, D)
    with pytest.raises(ValueError, match='for a Cox model'):
        GAM(TERMS, outcome=Outcome(Surv('time', 'status'))).fit(D, D)
    with pytest.raises(ValueError, match='Surv response'):
        GAM(TERMS, family=CoxFamily(), outcome=Outcome('time')).fit(D, D)
    with pytest.raises(ValueError, match='one column'):
        GAM(TERMS, outcome=Outcome(['y', 'z'])).fit(D, D)


def test_response_alone():
    # outcome=Surv(...) or Binomial(...): an Outcome of it
    D = _data()
    a = GAM(TERMS, family=CoxFamily(), outcome=Surv('time', 'status')).fit(D, D)
    b = GAM(TERMS, family=CoxFamily(), outcome=Outcome(Surv('time', 'status'))).fit(D, D)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(D, prediction_type='link'), rtol=1e-12)
    G = GAM(TERMS, family=BINOMIAL, outcome=Binomial('succ', trials='trials'))
    assert r'w_i &= \text{trials}_i' in G.model_latex()
    G.fit(D, D)
