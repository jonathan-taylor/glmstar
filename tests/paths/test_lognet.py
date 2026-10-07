from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


rng = np.random.default_rng(0)

from glmnet import LogNet

from .test_gaussnet import get_glmnet_soln

def get_RLogNet(Rinfo):
    RGLMNet = Rinfo["RGLMNet"]
    rpy = Rinfo["rpy"]
    @dataclass
    class RLogNet(RGLMNet):
        modified_newton: bool = False
        family: str= '"binomial"'
        def __post_init__(self):
            super().__post_init__()
            if self.modified_newton:
                rpy.r.assign("type.logistic", "modified.Newton")
            else:
                rpy.r.assign("type.logistic", "Newton")
            self.args["type.logistic"] = "type.logistic"
    return RLogNet

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


def test_lognet(Rinfo, modified_newton,
                penalty_factor,
                standardize,
                fit_intercept,
                n,
                p,
                sample_weight,
                offset,
                ):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)
        
    L = LogNet(modified_newton=modified_newton,
               standardize=standardize,
               fit_intercept=fit_intercept,
               penalty_factor=penalty_factor,
               **col_args)

    L.fit(X, D)

    C = get_glmnet_soln(Rinfo, get_RLogNet(Rinfo),
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

def test_CV(Rinfo, offset,
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

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

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
    C, CVM, CVSD = get_glmnet_soln(Rinfo, get_RLogNet(Rinfo),
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


@pytest.mark.parametrize('nobs, nvars, dfmax', [(100, 50, 3), (50, 200, 10)])
def test_lognet_df_max(Rinfo, nobs, nvars, dfmax):
    # df_max small enough that nx = min(2*df_max+20, p) < p, so that the
    # flattened coefficient array returned from C++ has stride nx, not p

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, Y, D, col_args, weightsR, offsetR = get_data(nobs, nvars, None, None)

    L = LogNet(df_max=dfmax, **col_args)
    L.fit(X, D)

    C = get_glmnet_soln(Rinfo, get_RLogNet(Rinfo),
                        X,
                        Y,
                        df_max=dfmax)

    assert C.shape[0] == L.coefs_.shape[0]
    assert np.linalg.norm(C[:,1:] - L.coefs_) / max(np.linalg.norm(L.coefs_), 1) < 1e-8
    assert np.linalg.norm(C[:,0] - L.intercepts_) / max(np.linalg.norm(L.intercepts_), 1) < 1e-8
