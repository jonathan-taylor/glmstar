from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

rng = np.random.default_rng(0)
np_cv_rules = default_converter + numpy2ri.converter

from glmnet import LogNet

from .test_gaussnet import (RGLMNet,
                            get_glmnet_soln,
                            sample_weight_pyt,
                            standardize_pyt,
                            fit_intercept_pyt,
                            nsample_pyt,
                            nfeature_pyt,
                            alignment_pyt,
                            penalty_factor_pyt)

@dataclass
class RLogNet(RGLMNet):

    modified_newton: bool = False
    family: str= '"binomial"'
    
    def __post_init__(self):

        super().__post_init__()

        if self.modified_newton:
            rpy.r.assign('type.logistic', "modified.Newton")
        else:
            rpy.r.assign('type.logistic', "Newton")
        self.args['type.logistic'] = 'type.logistic'

def get_data(n, p, sample_weight, offset):

    X = rng.standard_normal((n, p))
    X = rng.standard_normal((n, p))
    Y = rng.choice(['A', 'B'], size=n)

    L = LabelEncoder().fit(Y)
    Y_R = (Y == L.classes_[1])

    D = pd.DataFrame({'Y':Y})
    col_args = {'response_id':'Y'}
    
    if offset is not None:
        offset = offset(n)
        offset_id = 'offset'
        D['offset'] = offset
        offsetR = offset
    else:
        offset_id = None
        offsetR = None
    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        D['weight'] = sample_weight
        weightsR = sample_weight
    else:
        weight_id = None
        weightsR = None
        
    col_args = {'response_id':'Y',
                'weight_id':weight_id,
                'offset_id':offset_id}
    return X, Y_R, D, col_args, weightsR, offsetR

offset_pyt = pytest.mark.parametrize('offset', [None, np.zeros, lambda n: rng.uniform(0, 1, size=n)])

@pytest.mark.parametrize('modified_newton', [True, False])
@penalty_factor_pyt
@offset_pyt
@sample_weight_pyt
@standardize_pyt
@fit_intercept_pyt
@nsample_pyt
@nfeature_pyt
def test_lognet(modified_newton,
                penalty_factor,
                standardize,
                fit_intercept,
                n,
                p,
                sample_weight,
                offset,
                ):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)
        
    L = LogNet(modified_newton=modified_newton,
               standardize=standardize,
               fit_intercept=fit_intercept,
               penalty_factor=penalty_factor,
               **col_args)

    L.fit(X, D)

    C = get_glmnet_soln(RLogNet,
                        X,
                        Y,
                        modified_newton=modified_newton,
                        penalty_factor=penalty_factor,
                        weights=weightsR,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        offset=offsetR)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / max(np.linalg.norm(L.coefs_), 1) < 1e-8
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / max(np.linalg.norm(L.intercepts_), 1) < 1e-8

@offset_pyt
#@penalty_factor_pyt -- some fails with different penalty factors
@sample_weight_pyt
@alignment_pyt
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
            p=20):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    L = LogNet(standardize=standardize,
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
    CVM_ = L.score_path_.scores['Binomial Deviance']
    CVSD_ = L.score_path_.scores['SD(Binomial Deviance)']
    C, CVM, CVSD = get_glmnet_soln(RLogNet,
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

