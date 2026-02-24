from dataclasses import dataclass
import string
from copy import copy

import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

rng = np.random.default_rng(0)
np_cv_rules = default_converter + numpy2ri.converter

from glmnet import MultiClassNet

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
        n, q = Y.shape
        labels = ', '.join([f'"{l}"' for l in string.ascii_uppercase[:q]])
        rpy.r(f'''
colnames(Y) <- c({labels}) # depends on q
''')

        if 'offset' in parser.args:
            rpy.r(f'colnames(offset) <- c({labels})')

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
        cmd = '\n'.join([f'C{l} = as.matrix(C${l})' for l in string.ascii_uppercase[:q]])
        print(cmd)
        rpy.r(cmd)

        C = np.array([rpy.r(f'C{l}') for l in string.ascii_uppercase[:q]])
        if parser.foldid is not None:
            CVM = rpy.r('CVM')
            CVSD = rpy.r('CVSD')

    if parser.foldid is None:
        return C.T
    else:
        return C.T, CVM, CVSD

@dataclass
class RMultiClassNet(RGLMNet):

    family: str= '"multinomial"'

def get_data(n, p, q, sample_weight, offset):

    X = rng.standard_normal((n, p))
    labels = list(string.ascii_uppercase[:q])
    Y = rng.choice(labels, size=n)
    R = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape((-1,1)))
    
    Df = pd.DataFrame({'response':Y})
    response_id = 'response'

    if offset is not None:
        O = offset((n, q))
        offset_id = [f'offset[{i}]' for i in range(q)]
        for i, l in enumerate(offset_id):
            Df[l] = O[:,i]
        offsetR = O
    else:
        offset_id = None
        offsetR = None

    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        Df['weight'] = sample_weight
        weightsR = sample_weight
    else:
        weight_id = None
        weightsR = None
        
    col_args = {'response_id':response_id,
                'weight_id':weight_id,
                'offset_id':offset_id}

    return X, R, Df, col_args, weightsR, offsetR



def test_multiclassnet(standardize,
                       fit_intercept,
                       n,
                       p,
                       q,
                       sample_weight,
                       offset,
                       ):

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, q, sample_weight, offset)
        
    L = MultiClassNet(standardize=standardize,
                      fit_intercept=fit_intercept, 
                      **col_args)

    L.fit(X, D)

    C = get_glmnet_soln(RMultiClassNet,
                        X,
                        Y,
                        weights=weightsR,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        offset=offsetR)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-5
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-5

def test_CV(offset,
            sample_weight,
            alignment,
            penalty_factor,
            df_max,
            standardize,
            fit_intercept,
            exclude,
            nlambda,
            lambda_min_ratio,
            n,
            p,
            q):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, q, sample_weight, offset)

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    L = MultiClassNet(standardize=standardize,
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
    CVM_ = L.score_path_.scores['Multinomial Deviance']
    CVSD_ = L.score_path_.scores['SD(Multinomial Deviance)']
    C, CVM, CVSD = get_glmnet_soln(RMultiClassNet,
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


    print(CVM)
    print(np.asarray(CVM_))
    assert np.allclose(CVM[:15], CVM_.iloc[:15])
    assert np.allclose(CVSD[:15], CVSD_.iloc[:15])
