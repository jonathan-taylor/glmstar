import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
import statsmodels.api as sm

from glmnet.regularized_glm import RegGLM

from .test_gaussnet import (ifrpy,
                            has_rpy2)

from .common import (ifrpy,
                     has_rpy2,
                     glmnetR,
                     baseR,
                     statR,
                     np_cv_rules,
                     rpy)

rng = np.random.default_rng(0)


@ifrpy
def test_glmnet(standardize,
                fit_intercept,
                sample_weight,
                alpha,
                path,
                offset,
                n=1000,
                p=50):

    n, p = 1000, 50

    if sample_weight is None:
        sample_weight = np.ones(n)
    else:
        sample_weight = sample_weight(n)
        
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

        rpy.r.assign('X', X)
        rpy.r.assign('beta', beta)
        rpy.r.assign('sample_weight', sample_weight)
        rpy.r.assign('n', n)
        rpy.r.assign('p', p)
        rpy.r('y = rbinom(n, 1, 0.5)')
        rpy.r.assign('fit_intercept', fit_intercept)
        rpy.r.assign('standardize', standardize)
        rpy.r.assign('alpha', alpha)
        rpy.r.assign('offset', offset)
        rpy.r('''
        G = glmnet(X,
                   y,
                   weights=sample_weight,
                   intercept=fit_intercept,
                   standardize=standardize,
                   family=binomial(link="probit"),
                   offset=offset,
                   alpha=alpha)
        B = predict(G,
                    s=0.5/sqrt(n),
                    type="coef",
                    exact=TRUE,
                    x=X,
                    y=y,
                    offset=offset,
                    weights=sample_weight)
        ''')
        y = rpy.r('y')
        soln_R = rpy.r('as.numeric(B)')
        intercept_R = soln_R[0]
        coef_R = soln_R[1:]

    G = RegGLM(lambda_val=0.5/np.sqrt(n),
               family=sm.families.Binomial(sm.families.links.Probit()),
               alpha=alpha,
               standardize=standardize, 
               fit_intercept=fit_intercept,
               response_id='response',
               weight_id='weight',
               offset_id='offset')

    df = pd.DataFrame({'response':y, 'weight':sample_weight, 'offset':offset})
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

    
def test_cv(standardize,
            fit_intercept,
            n=1000,
            p=50):

    X = rng.standard_normal((n, p))
    X[:,4] *= 1.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]
    y = (rng.standard_normal(n) + X @ beta > 0)

    G = RegGLM(lambda_val=0.5 / np.sqrt(n),
               family=sm.families.Binomial(sm.families.links.Probit()),
               fit_intercept=fit_intercept,
               standardize=standardize)
    cross_validate(G, X, y, cv=5)





