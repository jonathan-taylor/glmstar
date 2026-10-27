# y named by role, Y = Outcome(...).fit_transform(D), for GAM and GLMM: the same fits as by
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
    Y = outcome.fit_transform(D)
    a = _gam(GAM, family).fit(D, Y)
    b = _gam(GAM, family, **ids).fit(R, R)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.edf_, b.edf_, rtol=1e-9)
    np.testing.assert_allclose(a.score(D, Y), b.score(R, R), rtol=1e-9)


@pytest.mark.parametrize('name', ['gaussian', 'poisson', 'binomial-trials', 'cox'])
def test_glmm_matches_ids(name):
    pytest.importorskip('jax')
    family, outcome, ids = CASES[name]
    D = _data()
    R = _ref_data(D)
    terms = [s('x'), random_effect('g')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExperimentalWarning)
        a = _gam(GLMM, family, terms).fit(D, outcome.fit_transform(D))
        b = _gam(GLMM, family, terms, **ids).fit(R, R)
    np.testing.assert_allclose(a.smoothing_.lam, b.smoothing_.lam, rtol=1e-6)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-6, atol=1e-9)


def test_feature_offset():
    # mgcv's gam(cnt ~ s(x) + z + offset(log(exposure)), family=poisson)
    D = _data()
    log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                                   name='log(exposure)')
    outcome = Outcome('cnt', offset=log_exposure)
    a = GAM(TERMS, family=POISSON).fit(D, outcome.fit_transform(D))
    R = D.assign(log_exposure=np.log(D['exposure']))
    b = GAM(TERMS, family=POISSON, response_id='cnt', offset_id='log_exposure').fit(R, R)
    np.testing.assert_allclose(a.predict(D, prediction_type='link'),
                               b.predict(R, prediction_type='link'), rtol=1e-9)
    offset = outcome.transform(D.iloc[:10], response=False)['offset']
    np.testing.assert_allclose(a.predict(D.iloc[:10], offset=offset),
                               b.predict(R.iloc[:10], offset=R['log_exposure'].iloc[:10]),
                               rtol=1e-9)
    assert r'\text{offset}_i' in a.model_latex()


def test_latex():
    # the offset and weights of y, once fitted; of the ids, also before
    D = _data()
    Y = Outcome(Binomial('succ', trials='trials', output='proportion'), offset='O').fit_transform(D)
    gam = GAM(TERMS, family=BINOMIAL)
    assert r'\text{offset}_i' not in gam.model_latex()
    gam.fit(D, Y)
    tex = gam.model_latex()
    assert r'\text{offset}_i' in tex and r'w_i &= \text{weight}_i' in tex
    ids = GAM(TERMS, offset_id='O', weight_id='w').model_latex()
    assert r'\text{O}_i' in ids and r'w_i &= \text{w}_i' in ids


def test_sklearn():
    D = _data()
    cv = KFold(4, shuffle=True, random_state=0)
    a = cross_val_score(GAM(TERMS, family=POISSON), D, Outcome('cnt', offset='O').fit_transform(D),
                        cv=cv)
    b = cross_val_score(GAM(TERMS, family=POISSON, response_id='cnt', offset_id='O'), D, D, cv=cv)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_errors():
    D = _data()
    with pytest.raises(ValueError, match='no columns stop and status'):
        GAM(TERMS, family=CoxFamily()).fit(D, Outcome('time').fit_transform(D))
    with pytest.raises(ValueError, match='one column'):
        GAM(TERMS).fit(D, Outcome(['y', 'z']).fit_transform(D))
