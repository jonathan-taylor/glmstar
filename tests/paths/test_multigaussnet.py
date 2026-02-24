from dataclasses import dataclass
from copy import copy

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import KFold

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

rng = np.random.default_rng(0)
np_cv_rules = default_converter + numpy2ri.converter

from glmnet import MultiGaussNet

from .test_gaussnet import RGLMNet

def get_glmnet_soln(parser_cls,
                    X,
                    Y,
                    **args):

    parser = parser_cls(**args)
    args, cvargs = parser.parse()
    
    with np_cv_rules.context():
        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
library(glmnet)
X = as.matrix(X)
G = glmnet(X, Y, {args})
C = coef(G)
if (doCV) {{
    foldid = as.integer(foldid)
    CVG = cv.glmnet(X, Y, {cvargs})
    CVM = CVG$cvm
    CVSD = CVG$cvsd
}}
            '''
        print(cmd)

        rpy.r(cmd)

        n, q = Y.shape
        cmd = '\n'.join([f'C{i} = as.matrix(C$y{i+1})' for i in range(q)])
        print(cmd)
        rpy.r(cmd)

        C = np.array([rpy.r(f'C{i}') for i in range(q)])
        if parser.foldid is not None:
            CVM = rpy.r('CVM')
            CVSD = rpy.r('CVSD')

    if parser.foldid is None:
        return C.T
    else:
        return C.T, CVM, CVSD

@dataclass
class RMultiGaussNet(RGLMNet):

    family: str= '"mgaussian"'
    

def get_data(n, p, q, sample_weight, offset):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))

    D = [Y]
    
    response_id = [f'response[{i}]' for i in range(q)]

    columns = copy(response_id)

    if offset is not None:
        offset = offset((n, q))
        offset_id = [f'offset[{i}]' for i in range(q)]
        offsetR = offset
        D.append(offset)
        columns.extend(offset_id)
    else:
        offset_id = None
        offsetR = None

    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        D.append(sample_weight)
        weightsR = sample_weight
        columns.append('weight')
    else:
        weight_id = None
        weightsR = None
        
    D = np.column_stack(D)
    Df = pd.DataFrame(D, columns=columns)
    col_args = {'response_id':response_id,
                'weight_id':weight_id,
                'offset_id':offset_id}

    return X, Y, Df, col_args, weightsR, offsetR



def test_mrelnet(standardize,
                 fit_intercept,
                 n,
                 p,
                 q,
                 sample_weight,
                 offset,
                 ):

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, q, sample_weight, offset)
        
    L = MultiGaussNet(standardize=standardize,
                      fit_intercept=fit_intercept, 
                      **col_args)

    L.fit(X, D)

    C = get_glmnet_soln(RMultiGaussNet,
                        X,
                        Y,
                        weights=weightsR,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        offset=offsetR)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / max(np.linalg.norm(L.coefs_), 1) < 1e-8
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / max(np.linalg.norm(L.intercepts_), 1) < 1e-8

def test_CV(offset,
            sample_weight,
            alignment,
            penalty_factor=None,
            df_max=None,
            standardize=True,
            fit_intercept=True,
            exclude=[],
            nlambda=None,
            lambda_min_ratio=None,
            n=103,
            p=20,
            q=3):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, q, sample_weight, offset)

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    L = MultiGaussNet(standardize=standardize,
                      fit_intercept=fit_intercept,
                      lambda_min_ratio=lambda_min_ratio,
                      penalty_factor=penalty_factor,
                      exclude=exclude,
                      df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)
    L.cross_validation_path(X,
                            D,
                            alignment=alignment,
                            cv=cv)
    CVM_ = L.score_path_.scores['Mean Squared Error']
    CVSD_ = L.score_path_.scores['SD(Mean Squared Error)']
    C, CVM, CVSD = get_glmnet_soln(RMultiGaussNet,
                                   X,
                                   Y.copy(),
                                   standardize=standardize,
                                   fit_intercept=fit_intercept,
                                   penalty_factor=penalty_factor,
                                   exclude=exclude,
                                   weights=weightsR,
                                   nlambda=nlambda,
                                   offset=offsetR,
                                   df_max=df_max,
                                   lambda_min_ratio=lambda_min_ratio,
                                   foldid=foldid,
                                   alignment=alignment)


    assert np.allclose(CVM, CVM_)
    assert np.allclose(CVSD, CVSD_)

