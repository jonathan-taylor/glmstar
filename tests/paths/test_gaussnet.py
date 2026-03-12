from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from copy import copy

import pytest
rng = np.random.default_rng(0)

from sklearn.model_selection import KFold


from glmnet import GaussNet
def get_RGaussNet(Rinfo):
    RGLMNet = Rinfo["RGLMNet"]
    @dataclass
    class RGaussNet(RGLMNet):
        covariance: bool = False
        def __post_init__(self):
            super().__post_init__()
            if self.covariance:
                self.args["type.gaussian"] = '"covariance"'
    return RGaussNet



def get_glmnet_soln(Rinfo,
                    parser_cls,
                    X,
                    Y,
                    **args):
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    parser = parser_cls(**args)
    args, cvargs = parser.parse()
    
    with np_cv_rules.context():
        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
library(glmnet)
Y = as.numeric(Y)
X = as.matrix(X)
G = glmnet(X, Y, {args})
C = as.matrix(coef(G))
if (doCV) {{
    foldid = as.integer(foldid)
    CVG = cv.glmnet(X, Y, {cvargs})
    CVM = CVG$cvm
    CVSD = CVG$cvsd
}}
            '''
        print(cmd)

        rpy.r(cmd)
        C = rpy.r('C')
        if parser.foldid is not None:
            CVM = rpy.r('CVM')
            CVSD = rpy.r('CVSD')

    if parser.foldid is None:
        return C.T
    else:
        return C.T, CVM, CVSD

# weight functions

def sample1(n):
    return rng.uniform(0, 1, size=n)

def sample2(n, num_zero=4):
    V = sample1(n)
    V[rng.choice(n, size=num_zero)] = 0
    return V

def get_data(n, p, sample_weight, offset):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
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
    return X, Y, D, col_args, weightsR, offsetR




def test_gaussnet(Rinfo, covariance,
                  standardize,
                  fit_intercept,
                  exclude,
                  lower_limits,
                  nlambda,
                  lambda_min_ratio,
                  sample_weight,
                  df_max):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')


    n, p = 500, 50

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, None)

    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 exclude=exclude,
                 lower_limits=lower_limits,
                 lambda_min_ratio=lambda_min_ratio,
                 df_max=df_max, **col_args)

    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(Rinfo, get_RGaussNet(Rinfo),
                        X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        exclude=exclude,
                        lower_limits=lower_limits,
                        weights=weightsR,
                        offset=offsetR,
                        nlambda=nlambda,
                        lambda_min_ratio=lambda_min_ratio,
                        df_max=df_max)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10


def test_limits(Rinfo, limits,
                penalty_factor,
                sample_weight,
                df_max=None,
                covariance=None,
                standardize=True,
                fit_intercept=True,
                exclude=[],
                nlambda=None,
                lambda_min_ratio=None,
                n=100,
                p=50):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    lower_limits, upper_limits = limits
    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits
    if upper_limits is not None:
        upper_limits = np.ones(p) * upper_limits
    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, None)

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 upper_limits=upper_limits,
                 lower_limits=lower_limits,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(Rinfo, get_RGaussNet(Rinfo),
                        X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weightsR,
                        nlambda=nlambda,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    tol = 1e-10
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

def test_offset(Rinfo, offset,
                penalty_factor,
                sample_weight,
                df_max=None,
                covariance=None,
                standardize=True,
                fit_intercept=True,
                exclude=[],
                nlambda=None,
                lambda_min_ratio=None,
                n=100,
                p=50):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(Rinfo, get_RGaussNet(Rinfo),
                        X,
                        Y.copy(),
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weightsR,
                        nlambda=nlambda,
                        offset=offsetR,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    tol = 1e-10
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

def test_CV(Rinfo, offset,
            penalty_factor,
            sample_weight,
            alignment,
            df_max=None,
            covariance=None,
            standardize=True,
            fit_intercept=True,
            exclude=[],
            nlambda=None,
            lambda_min_ratio=None,
            n=103,
            p=50):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
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
    C, CVM, CVSD = get_glmnet_soln(Rinfo, get_RGaussNet(Rinfo),
                                   X,
                                   Y.copy(),
                                   covariance=covariance,
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


def test_prefilter_excludes_features(Rinfo, n, p):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    from glmnet.paths.gaussnet import GaussNet

    class PrefilterGaussNet(GaussNet):
        def prefilter(self, X, y):
            # Exclude features where the sum of X[:, j] * y is negative
            print(np.nonzero(X.T @ y > 0)[0].shape)
            return np.nonzero(X.T @ y > 0)[0]

    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    X[:,:2] *= np.sign(X.T @ y)[:2][None,:]
    model = PrefilterGaussNet()
    model.fit(X, y)

    # All excluded features should have all-zero coefficients for all lambdas
    excluded = model.excluded_
    assert excluded.shape, "No features were excluded by prefilter"
    coefs = model.coefs_
    assert np.allclose(coefs[:, excluded], 0)


def test_prefilter_and_explicit_exclude(Rinfo):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    from glmnet.paths.gaussnet import GaussNet

    class PrefilterGaussNet(GaussNet):
        def prefilter(self, X, y):
            # Exclude features where the sum of X[:, j] * y is negative
            return np.nonzero(X.T @ y > 0)[0]

    X = rng.standard_normal((100, 10))
    y = rng.standard_normal(100)
    X[:,:2] *= -np.sign(X.T @ y)[:2][None,:]

    # Explicitly exclude feature 0, and let prefilter exclude others
    model = PrefilterGaussNet(exclude=[0])
    model.fit(X, y)
    excluded = model.excluded_
    assert 0 in excluded, "Explicitly excluded feature 0 not in exclude list"
    coefs = model.coefs_
    assert np.allclose(coefs[:, excluded], 0)

