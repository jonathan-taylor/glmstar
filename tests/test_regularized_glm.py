from dataclasses import asdict

import pytest

import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm

from glmnet.elnet import ElNet
from glmnet.regularized_glm import RegGLM

rng = np.random.default_rng(0)


def test_compare_regglm_elnet(standardize,
                              fit_intercept,
                              sample_weight,
                              lambda_val,
                              alpha):

    sample_weight = sample_weight(n)
    
    X = rng.normal(size=(n,p))
    coefs = np.zeros(p)
    coefs[[2,3]] = 3 / np.sqrt(n)
    y = rng.normal(size=n) + 1 + X @ coefs

    df = pd.DataFrame({'response':y,
                       'weight':sample_weight})

    elnet = ElNet(lambda_val,
                  alpha=alpha,
                  standardize=standardize,
                  fit_intercept=fit_intercept)
    elnet.fit(X, 
              y,
              sample_weight=sample_weight / sample_weight.mean())

    reg_glm = RegGLM(lambda_val / n,
                     alpha=alpha,
                     standardize=standardize,
                     fit_intercept=fit_intercept,
                     weight_id='weight')
    reg_glm.fit(X, 
               df)

    if not np.allclose(elnet.intercept_, reg_glm.intercept_, rtol=1e-4):
        raise ValueError('intercepts not close')
    if not np.linalg.norm(elnet.coef_ - reg_glm.coef_) / np.linalg.norm(reg_glm.coef_) < 1e-4:
        raise ValueError('coefs not close')

    elnet_dict = asdict(elnet)
    reg_glm_dict = asdict(reg_glm)

    eta = X @ reg_glm.coef_ + reg_glm.intercept_
    dev = np.sum(sample_weight * (y - eta)**2) 
    # dev_ratio = 1 - dev / reg_glm_dict['nulldev']

    # # check the regularizer 
    # assert(dev_ratio == glmnet_dict['dev_ratio'])
    
    # del(elnet_dict['a0']) # compared above
    # del(elnet_dict['beta']) # compared above
    # del(elnet_dict['warm_fit']) # don't compare
    # del(elnet_dict['npasses'])
    
    # failures = []

    # for k in elnet_dict:
    #     try:
    #         elnet_v = np.asarray(elnet_dict[k])
    #         glmnet_v = np.asarray(glmnet_dict[k])
    #     except:
    #         elnet_v = glmnet_v = 0
    #     if not np.allclose(elnet_v, glmnet_v):
    #         failures.append(f'field {k} differs {elnet_v}, {glmnet_v}')

    # if failures:
    #     raise ValueError(';'.join(failures))

def test_compare_sparse_elnet(standardize,
                              fit_intercept,
                              sample_weight,
                              lambda_val):

    sample_weight = sample_weight(n)
    X = rng.normal(size=(n,p))
    y = rng.normal(size=n)

    elnet = ElNet(lambda_val,
                           standardize=standardize,
                           fit_intercept=fit_intercept)
    elnet.fit(X, 
              y,
              sample_weight)

    Xs = scipy.sparse.csc_array(X).tocsc()
    elnet_s = ElNet(lambda_val,
                             standardize=standardize,
                             fit_intercept=fit_intercept)
    elnet_s.fit(Xs, 
                y,
                sample_weight)

    if not np.allclose(elnet.intercept_, elnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(elnet.coef_, elnet_s.coef_):
        raise ValueError('coefs not close')

def test_compare_sparse_glmnet(standardize,
                               fit_intercept,
                               sample_weight,
                               lambda_val):

    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    beta = np.zeros(p)
    beta[:2] = [2,-2.3]
    y = rng.normal(size=n) + X @ beta
    sample_weight = sample_weight(n)

    glmnet = RegGLM(lambda_val,
                    standardize=standardize,
                    fit_intercept=fit_intercept)
    df = pd.DataFrame({'response':y})
    glmnet.fit(X,
               df,
               sample_weight=sample_weight)

    Xs = scipy.sparse.csc_array(X).tocsc()
    glmnet_s = RegGLM(lambda_val,
                      standardize=standardize,
                      fit_intercept=fit_intercept)
    glmnet_s.fit(Xs,
                 y,
                 sample_weight=sample_weight)

    if not np.allclose(glmnet.intercept_, glmnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(glmnet.coef_, glmnet_s.coef_):
        raise ValueError('coefs not close')
    
def test_logistic(standardize,
                  fit_intercept,
                  sample_weight,
                  lambda_val):

    n, p = 100, 10
    sample_weight = sample_weight(n)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    glmnet = RegGLM(lambda_val,
                    standardize=standardize,
                    fit_intercept=fit_intercept,
                    family=sm.families.Binomial())
    df = pd.DataFrame({'response':y})
    glmnet.fit(X, 
               df,
               weights)
    print(glmnet.coef_)

def test_probit(standardize,
                fit_intercept,
                sample_weight,
                lambda_val):

    n, p = 100, 10
    sample_weight = sample_weight(n)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    link = sm.families.links.Probit()
    glmnet = RegGLM(lambda_val,
                    standardize=standardize,
                    fit_intercept=fit_intercept,
                    family=sm.families.Binomial(link=link))
    df = pd.DataFrame({'response':y})
    glmnet.fit(X, 
               df,
               weights)
    print(glmnet.coef_)
