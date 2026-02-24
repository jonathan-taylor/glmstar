import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
import statsmodels.api as sm

from glmnet.cox import (RegCoxLM,
                        CoxFamilySpec,
                        CoxNet)
from glmnet.data import make_survival

from .test_gaussnet import (ifrpy,
                            standardize,
                            fit_intercept,
                            sample_weight,
                            alpha,
                            path,
                            offset,
                            has_rpy2)

if has_rpy2:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter
    import rpy2.robjects as rpy
    
    np_cv_rules = default_converter + numpy2ri.converter

    glmnetR = importr('glmnet')
    baseR = importr('base')
    statR = importr('stats')
    importr('survival')
    
rng = np.random.default_rng(0)


@ifrpy
@standardize
@sample_weight
@alpha
@path
@offset
def test_glmnet(standardize,
                sample_weight,
                alpha,
                path,
                offset,
                n=831,
                p=8):

    if sample_weight is None:
        sample_weight = np.ones(n)
    else:
        sample_weight = sample_weight(n)
        
    status = rng.choice([0, 1], size=n)
    start = rng.integers(0, 5, size=n)
    event = start + rng.integers(1, 5, size=n)
    event_data = pd.DataFrame({'event':event,
                               'status':status,
                               'start':start})

    family = CoxFamilySpec(event_data,
                           tie_breaking='breslow',
                           event_id='event',
                           status_id='status',
                           start_id='start')
    
    X = rng.standard_normal((n, p))
    X[:,4] *= 4.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]

    if offset:
        offset = rng.standard_normal(n) * 0.3
    else:
        offset = np.zeros(n)

    with np_cv_rules.context():

        rpy.r('library(survival)')
        rpy.r.assign('X', X)
        rpy.r.assign('beta', beta)
        rpy.r.assign('sample_weight', sample_weight)
        rpy.r.assign('n', n)
        rpy.r.assign('p', p)
        rpy.r.assign('start', start)
        rpy.r.assign('event', event)
        rpy.r.assign('status', status)
        rpy.r('sample_weight=as.numeric(sample_weight)')
        rpy.r('start=as.numeric(start)')
        rpy.r('event=as.numeric(event)')
        rpy.r('status=as.integer(status)')
        rpy.r('y = Surv(start, event, status)')
        rpy.r.assign('standardize', standardize)
        rpy.r.assign('alpha', alpha)
        rpy.r.assign('offset', offset)
        rpy.r('offset=as.numeric(offset)')
        rpy.r('''
        G = glmnet(X,
                   y,
                   weights=sample_weight,
                   standardize=standardize,
                   family='cox',
                   offset=offset,
                   alpha=alpha)
        ''')
        rpy.r('''
        B = predict(G,
                    s=0.5/sqrt(n),
                    type="coef",
                    exact=TRUE,
                    x=X,
                    y=y,
                    offset=offset,
                    weights=sample_weight)
        ''')
        soln_R = rpy.r('as.numeric(B)')
        intercept_R = 0
        coef_R = soln_R

    G = RegCoxLM(lambda_val=0.5/np.sqrt(n),
                 family=family,
                 alpha=alpha,
                 standardize=standardize, 
                 weight_id='weight',
                 offset_id='offset')

    df = pd.DataFrame({'start':start, 'event':event, 'status':status, 'weight':sample_weight, 'offset':offset})
    G.fit(X, df)

    soln_py = np.hstack([G.intercept_, G.coef_])
    soln_R = np.hstack([intercept_R, coef_R])    
    yhat_py = G.design_ @ soln_py
    yhat_R = G.design_ @ soln_R 

    fit_match = np.allclose(yhat_py, yhat_R, atol=1e-3, rtol=1e-3)
    intercept_match = np.fabs(G.intercept_ - intercept_R) < max(np.fabs(intercept_R), 1) * 1e-3
    coef_match = np.allclose(G.coef_, coef_R, atol=1e-3, rtol=1e-3)

    print(f'fit: {fit_match}, intercept: {intercept_match}, coef:{coef_match}')

    assert fit_match and intercept_match and coef_match


def test_stratified_cox_agrees_with_unstratified():
    """
    When strata_id is None or all values are the same, stratified and unstratified Cox should agree.
    """
    n, p = 100, 5
    X, y, coef = make_survival(n_samples=n, n_features=p, random_state=42)
    y['strata'] = 0  # Single stratum

    # Unstratified
    fam_unstrat = CoxFamilySpec(y, event_id='event', status_id='status', strata_id=None)
    fit_unstrat = CoxNet(family=fam_unstrat).fit(X, y)
    # Stratified, but only one stratum
    fam_strat = CoxFamilySpec(y, event_id='event', status_id='status', strata_id='strata')
    fit_strat = CoxNet(family=fam_strat).fit(X, y)

    # Compare coefficient paths
    assert np.allclose(fit_unstrat.coefs_, fit_strat.coefs_, atol=1e-8)
    assert np.allclose(fit_unstrat.intercepts_, fit_strat.intercepts_, atol=1e-8)

    # Now test with strata_id=None (should be same as above)
    fam_none = CoxFamilySpec(y, event_id='event', status_id='status', strata_id=None)
    fit_none = CoxNet(family=fam_none).fit(X, y)
    assert np.allclose(fit_unstrat.coefs_, fit_none.coefs_, atol=1e-8)


def test_stratified_cox_differs_with_multiple_strata():
    """
    When strata_id has multiple unique values, stratified and unstratified Cox should differ (unless by chance).
    """
    n, p = 100, 5
    X, y, coef = make_survival(n_samples=n, n_features=p, random_state=42)
    # Create 3 strata
    y['strata'] = np.repeat(np.arange(3), n // 3 + 1)[:n]

    fam_unstrat = CoxFamilySpec(y, event_id='event', status_id='status', strata_id=None)
    fit_unstrat = CoxNet(family=fam_unstrat).fit(X, y)
    fam_strat = CoxFamilySpec(y, event_id='event', status_id='status', strata_id='strata')
    fit_strat = CoxNet(family=fam_strat).fit(X, y)

    # The coefficient paths should generally differ
    assert not np.allclose(fit_unstrat.coefs_[:20], fit_strat.coefs_[:20], atol=1e-6)






